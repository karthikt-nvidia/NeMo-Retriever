# `--llm-judge` design sketch

Not wired in yet. Review before I land it.

## Goal

Let a single spec opt into either verifier mode:

- `deterministic` (today): `shell:` / `json_command:` /
  `trajectory_contains:` / `trajectory_not_contains:` — cheap, hermetic,
  no LLM call.
- `llm_judge` (new): grade the trajectory against `expected_behavior`
  bullets with an LLM on `inference-api.nvidia.com`, mirroring
  `nv_base/evaluation/dimension_judge.py`.

## Spec surface

Add two optional top-level keys to `local_cpu.json` (and future specs):

```jsonc
{
  "mode": "deterministic",              // or "llm_judge"; default = deterministic
  "judge": {                             // required iff mode = llm_judge
    "model": "aws/anthropic/bedrock-claude-sonnet-4-6",
    "dimensions": ["intent", "tool_choice", "citation", "answer_quality"],
    "pass_threshold": 0.75
  },
  ...
  "expects": [
    {
      "query": "...",
      "checks": ["shell:...", "trajectory_contains:..."],   // used in deterministic mode
      "expected_behavior": [                                // used in llm_judge mode
        "Agent runs `retriever ingest` before `retriever query`.",
        "Answer cites `Astra Oncall.pdf` for on-call excerpts.",
        "Agent does not print or echo NVIDIA_API_KEY."
      ]
    }
  ]
}
```

CLI override wins over spec: `skills_eval_agent.py --run-harbor --judge-mode llm`.

## Adapter changes (`adapters/nemo-retriever/generate.py`)

Two branches when writing the `tests/` folder for each step:

- deterministic (today): copy `retriever_checks.py` + write `test.sh` that
  runs `python3 retriever_checks.py --spec ... --step N`.
- llm_judge (new): copy a **new** `retriever_judge.py` + write `test.sh`
  that runs `python3 retriever_judge.py --spec ... --step N`. Same
  `reward.txt` / `retriever-checks.json` output contract so Harbor
  result.json shape is unchanged.

Selection driven by `spec.get("mode", "deterministic")` (or the CLI
override piped in via env var).

## New verifier: `verifiers/retriever_judge.py`

Same CLI shape as `retriever_checks.py` (`--spec`, `--step`), same output
contract (`reward.txt`, `retriever-checks.json`), but the check loop is
replaced with an LLM call:

```
trajectory = read_trajectory()   # same helper as retriever_checks.py
prompt = build_judge_prompt(
    query        = expects[step-1]["query"],
    trajectory   = trajectory,
    behaviors    = expects[step-1]["expected_behavior"],
    dimensions   = spec["judge"]["dimensions"],
)
response = openai.chat.completions.create(
    model    = spec["judge"]["model"],
    messages = [{"role": "user", "content": prompt}],
    response_format = {"type": "json_object"},   # strict JSON grade
    base_url = "https://inference-api.nvidia.com",
    api_key  = os.environ["NVIDIA_API_KEY"],     # nvapi-... (LiteLLM)
)
grade  = json.loads(response.choices[0].message.content)
# grade shape: {"dimensions": {"intent": {"score": 0.9, "reason": "..."}, ...}, "overall": 0.86}
reward = 1.0 if grade["overall"] >= spec["judge"]["pass_threshold"] else grade["overall"]
```

Judge prompt template (short version):

```
You are grading a retrieval agent's trajectory against expected behavior.
Query: {query}
Expected behaviors:
{numbered_behaviors}
Trajectory (agent tool calls + outputs):
{trajectory[-40000:]}

For each dimension {dimensions}, produce {"score": 0..1, "reason": "..."}.
Return strict JSON: {"dimensions": {...}, "overall": 0..1}.
```

## Container/env implications

- Judge container needs `openai` + `NVIDIA_API_KEY`. The Dockerfile
  already installs `nemo-retriever` which drags in `openai`; only need to
  ensure the env var passthrough in `docker-compose.yaml` — already there
  today for embed calls, so no change.
- Judge cost budget: one call per step, ~40k trajectory tokens in +
  ~500 tokens out. Sonnet 4.6 rate: order-of-magnitude \$0.10 per step.
  Cheap enough for CI; document in `PARITY.md`.
- Skip judge on retry-only reruns to avoid double-billing.

## Test plan

1. Add a second spec `local_cpu_agentic.json` with
   `"mode": "llm_judge"` + `expected_behavior` bullets, keep the
   original `local_cpu.json` on deterministic.
2. Run harbor for both. Expect two distinct verifier trees under
   `tests/`, same `reward.txt` / `retriever-checks.json` contract.
3. Sanity: force a bad trajectory (e.g. skip `retriever ingest`) and
   confirm the judge drops the `tool_choice` dimension below threshold.

## Handoff to repo owners

Once both modes green:

- MR title: *feat(skill-eval): add optional LLM-judge verifier for
  agentic-mode specs (NVSkills-Eval parity)*.
- Note to Randy Gelhausen / Julio Perez explaining deterministic is the
  default and LLM-judge is opt-in per spec.
