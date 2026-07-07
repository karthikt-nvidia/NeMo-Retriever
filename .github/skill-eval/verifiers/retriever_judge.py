#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-as-judge verifier for nemo-retriever skill eval tasks.

Mirrors nv_base/evaluation/dimension_judge.py: grades the agent trajectory
against `expected_behavior` bullets on a per-dimension 0..1 scale using
inference-api.nvidia.com (OpenAI-compatible). Same output contract as
retriever_checks.py: writes reward.txt + retriever-checks.json to
$NEMO_RETRIEVER_EVAL_VERIFIER_LOG_DIR (default /logs/verifier).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

TRAJECTORY_CANDIDATES = (
    "/logs/agent/trajectory.jsonl",
    "/logs/agent/trajectory.json",
    "/logs/agent/claude-code.txt",
    "/logs/agent/agent.log",
)

DEFAULT_DIMENSIONS = ["intent", "tool_choice", "citation", "answer_quality"]
DEFAULT_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-6"
DEFAULT_THRESHOLD = 0.75


def locate_trajectory() -> Path | None:
    override = os.environ.get("NEMO_RETRIEVER_EVAL_TRAJECTORY")
    if override:
        path = Path(override)
        return path if path.exists() else None
    for candidate in TRAJECTORY_CANDIDATES:
        path = Path(candidate)
        if path.exists():
            return path
    return None


def read_trajectory() -> str:
    path = locate_trajectory()
    if path is None:
        return ""
    return path.read_text(errors="replace")


def build_prompt(query: str, behaviors: list[str], dimensions: list[str], trajectory: str) -> str:
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(behaviors)) or "(none provided)"
    dims = ", ".join(dimensions)
    tail = trajectory[-40000:] if trajectory else "(empty)"
    return (
        "You are a strict evaluator grading a retrieval agent's trajectory.\n"
        f"User query: {query}\n\n"
        "Expected behaviors:\n"
        f"{numbered}\n\n"
        f"Grade each dimension from this list on a 0..1 scale: {dims}.\n"
        "For each dimension include a short reason. Also produce an overall 0..1 score\n"
        "as the weighted average of the dimensions. Be conservative: penalise missing\n"
        "citations, wrong tool choice, or leaked secrets.\n\n"
        "Return STRICT JSON only, no prose, matching this schema exactly:\n"
        '{"dimensions": {"<dim>": {"score": <float>, "reason": "<string>"}, ...},\n'
        ' "overall": <float>}\n\n'
        "Agent trajectory (tail):\n"
        "----- BEGIN TRAJECTORY -----\n"
        f"{tail}\n"
        "----- END TRAJECTORY -----\n"
    )


def call_judge(prompt: str, model: str) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(f"openai package missing in judge container: {exc}") from exc

    # inference-api.nvidia.com (LiteLLM gateway) expects `sk-...`; the
    # container's NVIDIA_API_KEY is the `nvapi-...` embed-NIM cred. The
    # harbor `--ae` layer already forwards ANTHROPIC_API_KEY set by the
    # outer wrapper to the LiteLLM key, so use that. Allow an explicit
    # override via NEMO_RETRIEVER_JUDGE_API_KEY.
    api_key = (
        os.environ.get("NEMO_RETRIEVER_JUDGE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "no judge API key set (NEMO_RETRIEVER_JUDGE_API_KEY / ANTHROPIC_API_KEY / NVIDIA_API_KEY)"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "NEMO_RETRIEVER_JUDGE_BASE", "https://inference-api.nvidia.com/v1"
        ),
    )
    # LiteLLM proxies to Bedrock don't always honor response_format
    # (silently return plain-text or empty). Try structured first, fall
    # back to unstructured + regex-extract of the first {...} block.
    def _create(with_response_format: bool):
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        if with_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    raw_content = ""
    for attempt in (True, False):
        try:
            response = _create(with_response_format=attempt)
        except Exception as exc:  # noqa: BLE001
            print(f"JUDGE_ATTEMPT_ERROR (response_format={attempt}): {exc}", file=sys.stderr)
            continue
        raw_content = (response.choices[0].message.content or "").strip()
        if raw_content:
            break

    log_dir = Path(os.environ.get("NEMO_RETRIEVER_EVAL_VERIFIER_LOG_DIR", "/logs/verifier"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "judge-raw.txt").write_text(raw_content or "(empty)")

    if not raw_content:
        raise ValueError("judge model returned empty content")

    # Direct parse; else pull first JSON object from wrapped text.
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            raise ValueError(f"judge reply is not JSON: {raw_content[:200]!r}")
        return json.loads(match.group(0))


def to_check_rows(grade: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dim, payload in (grade.get("dimensions") or {}).items():
        score = float(payload.get("score", 0.0))
        rows.append(
            {
                "check": f"llm_judge:{dim}",
                "pass": score >= threshold,
                "route": "llm_judge",
                "score": score,
                "evidence": payload.get("reason", ""),
            }
        )
    overall = float(grade.get("overall", 0.0))
    rows.append(
        {
            "check": "llm_judge:overall",
            "pass": overall >= threshold,
            "route": "llm_judge",
            "score": overall,
            "evidence": f"weighted overall vs threshold {threshold}",
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    expects = spec.get("expects") or []
    if args.step < 1 or args.step > len(expects):
        raise SystemExit(f"step {args.step} out of range for {len(expects)} expects")

    step = expects[args.step - 1]
    behaviors = step.get("expected_behavior") or []
    query = step.get("query", "")

    judge_cfg = spec.get("judge") or {}
    dimensions = judge_cfg.get("dimensions") or DEFAULT_DIMENSIONS
    model = judge_cfg.get("model") or DEFAULT_MODEL
    threshold = float(judge_cfg.get("pass_threshold", DEFAULT_THRESHOLD))

    trajectory = read_trajectory()
    prompt = build_prompt(query, behaviors, dimensions, trajectory)

    try:
        grade = call_judge(prompt, model)
    except Exception as exc:  # noqa: BLE001 -- report cleanly, don't crash harbor
        print(f"JUDGE_ERROR: {exc}", file=sys.stderr)
        grade = {
            "dimensions": {dim: {"score": 0.0, "reason": f"judge failed: {exc}"} for dim in dimensions},
            "overall": 0.0,
        }

    results = to_check_rows(grade, threshold)
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    overall = next((r["score"] for r in results if r["check"] == "llm_judge:overall"), 0.0)

    log_dir = Path(os.environ.get("NEMO_RETRIEVER_EVAL_VERIFIER_LOG_DIR", "/logs/verifier"))
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "reward.txt").write_text(f"{overall:.6f}\n")
    (log_dir / "retriever-checks.json").write_text(json.dumps(results, indent=2) + "\n")
    (log_dir / "judge-grade.json").write_text(json.dumps(grade, indent=2) + "\n")

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"{status} ({r['score']:.2f}): {r['check']} — {r['evidence']}")
    print(f"=== LLM-judge overall: {overall:.3f} (threshold {threshold}); {passed}/{total} dims pass ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
