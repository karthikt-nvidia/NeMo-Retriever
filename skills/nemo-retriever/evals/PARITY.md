# NVSkills-Eval Parity

Status as of 2026-07-15. Owned by `karthikt@nvidia.com` (SRE).

## What we ship today

- **1 spec, 1 query, 5 expected-behaviour checks** (`local_cpu.json`).
- **LLM-as-judge verifier** (`.github/skill-eval/verifiers/retriever_judge.py`)
  — grades the agent trajectory against each `expected_behavior` bullet,
  emits per-check `pass` + `matched` snippet + `rationale` + `cost_usd`.
- **Judge model:** `aws/anthropic/bedrock-claude-sonnet-4-6` via
  `inference-api.nvidia.com` (LiteLLM gateway). Overridable via the spec's
  `judge.model` field.
- **Corpus:** 3 AIQ PDFs synthesized at CI time (reportlab) under
  `fixtures/research_reports/` — text-only, gitignored, deterministic.
- **Runtime:** ~3 min end-to-end on CPU-only against hosted embed NIMs on
  `integrate.api.nvidia.com` (no local GPU).
- **Reward:** `passed / total` across the behaviour list — same shape RAG's
  `generic_judge.py` emits so Sarath's dashboard renders our card identically.

## Dashboard contract

`.github/workflows/skill-eval-harbor.yml` uploads results under
`nemo-retriever-skill-eval-results-<run_id>`. The card at
`bp-skill-eval-dashboard.nvidia.com` polls that artifact name via the GH
Actions artifact API. Per-check drawer state is driven by `judge.json`
(`checks[]` with `pass`, `matched`, `rationale`, `cost_usd`).

## Divergence from NVSkills-Eval (upstream)

NVSkills-Eval (`NVIDIA/skills` → `nvcarps/ci-group/nvcarps-ci` →
`nv-base/evaluation/{dimension,insights}_judge.py`) runs the same LLM-judge
shape but ships a heavier stack (nv-base image + astra-skill-eval-cli +
GitLab runner). We intentionally stay on Harbor + public GH Actions.

## TODOs

1. **Expand spec coverage** — one spec per retrieval surface as it stabilises
   (dense, hybrid, agentic, rerank). Today: single on-call query.
2. **Move corpus out of gitignore.** Either check in a tiny redistributable
   fixture, or fetch from a canonical NGC/PBSS bucket at task-build time.
3. **Cost budget doc.** Publish per-run token/cost expectations so repo
   owners can wire CI budgets.

## Non-goals

- Not porting nv-base or astra-skill-eval-cli into this repo.
- Not requiring an internal GitLab runner. Public GH Actions only.
