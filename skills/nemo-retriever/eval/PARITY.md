# NVSkills-Eval Parity — TODO

Status as of 2026-07-07. Owned by `karthikt@nvidia.com` (SRE).

## What we ship today

- **1 spec, 1 query, 4 deterministic checks** (`local_cpu.json`).
- **Deterministic verifier only** (`shell:` / `trajectory_contains:` /
  `trajectory_not_contains:` / `json_command:`) via
  `.github/skill-eval/verifiers/retriever_checks.py`.
- **Corpus:** 3 AIQ PDFs under `fixtures/research_reports/` (gitignored, dev
  copies only — see `~/knowledge/skill-evals.md`).
- **Runtime:** ~3 min end-to-end on CPU-only against hosted embed NIMs on
  `integrate.api.nvidia.com` (no local GPU).
- **Reward:** 1.0 on green path.

## What NVSkills-Eval does that we do not — yet

NVSkills-Eval (`NVIDIA/skills` → `NVIDIA/nvskills-ci` → GitLab
`nvcarps/ci-group/nvcarps-ci` → `nv-base` + `astra-skill-eval-cli`) evaluates
skill trajectories with an **LLM-as-judge**:

- `nv_base/evaluation/dimension_judge.py`
- `nv_base/evaluation/insights_judge.py`

The judge grades the trajectory against `expected_behavior` bullets (intent,
tool choice, evidence citation) rather than diffing shell output. This is
tolerant of the open-ended nature of agentic-mode retrieval — which is exactly
what NRL is landing on `main`:

- #2018 (MERGED) — Adding Agentic Retrieval as a new retrieval mode
- #2267 (MERGED) — Adds Agentic Retrieval into harness
- #1936 (OPEN) — Adding agentic eval pipeline
- #2274 (OPEN) — Dev/mahikaw/agentic enhancements
- #1784 (MERGED) — AbstractOperators for agentic patterns

Deterministic checks cannot distinguish "the agent picked the right retrieval
mode and merged partial evidence correctly" from "the agent stumbled through
the right shell commands by luck". LLM-judge can.

## TODOs to reach parity

1. **Add `--llm-judge` mode** to the harbor adapter so a spec can opt into
   LLM-graded runs against `expected_behavior` bullets, using
   `inference-api.nvidia.com` for the judge model.
2. **Expand spec coverage** beyond the single on-call query — one spec per
   retrieval mode (dense / hybrid / agentic) once the agentic surface
   stabilizes.
3. **Move corpus out of gitignore.** Either check in a tiny redistributable
   fixture, or fetch from a canonical NGC/PBSS bucket at task-build time.
   Today the PDFs live only on developer workstations.
4. **Judge-model choice + cost budget.** Pick a default (Sonnet 4.6 via
   inference-api) and document per-run token/cost expectations so repo owners
   can wire budgets in CI.
5. **Reveal both modes to repo owners.** Draft note to Randy Gelhausen /
   Julio Perez explaining the deterministic vs LLM-judge split and letting
   them opt each spec into either mode.

## Non-goals

- Not porting the full nv-base image or astra-skill-eval-cli into this repo.
  We stay on Harbor.
- Not requiring an internal GitLab runner. This CI runs on the public
  NVIDIA/NeMo-Retriever GitHub Actions runners.
