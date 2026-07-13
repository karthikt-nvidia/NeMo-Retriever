#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-as-judge verifier for nemo-retriever skill eval tasks.

Modeled on RAG's generic_judge.py (NVIDIA-AI-Blueprints/rag) — same
per-check output shape so the shared dashboard parser renders fork rows
identically to RAG rows.

Given the eval spec's `expected_behavior` list, asks a judge model to grade
each behavior independently as pass/fail with a supporting trajectory
snippet and one-sentence rationale. Writes:

  /logs/verifier/reward.txt   — single float: passed / total (0.0-1.0)
  /logs/verifier/judge.json   — {spec, step, query, total, passed, reward,
                                 trajectory_path, trajectory_found, checks:[]}
  /logs/verifier/judge-raw.txt — raw judge model reply (debug only)

Each check row: {check, pass, route:"agent", matched, rationale} — same keys
RAG emits.
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

DEFAULT_MODEL = "aws/anthropic/bedrock-claude-sonnet-4-6"


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


def build_prompt(query: str, behaviors: list[str], trajectory: str) -> str:
    numbered = "\n".join(f"{i + 1}. {b}" for i, b in enumerate(behaviors)) or "(none provided)"
    tail = trajectory[-40000:] if trajectory else "(empty)"
    return (
        "You are a strict evaluator grading a retrieval agent's trajectory.\n\n"
        f"User query: {query}\n\n"
        "Grade each of these expected behaviors independently. For each,\n"
        "decide pass/fail based only on evidence in the trajectory below.\n"
        "Be conservative: penalise missing citations, wrong tool choice, or\n"
        "leaked secrets.\n\n"
        "Expected behaviors:\n"
        f"{numbered}\n\n"
        "For each behavior, output:\n"
        '  - "pass":      true if the agent clearly did this, false otherwise\n'
        '  - "matched":   a short exact snippet from the trajectory that\n'
        "                 supports pass (or empty string if fail / no evidence)\n"
        '  - "rationale": one or two sentences justifying the verdict\n\n'
        "Return STRICT JSON only, no prose, matching this schema exactly:\n"
        '{"checks": [\n'
        '  {"pass": <bool>, "matched": "<string>", "rationale": "<string>"},\n'
        "  ...\n"
        "]}\n\n"
        "The array MUST have the same length and order as the numbered list above.\n\n"
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
    # harbor `--ae` layer forwards ANTHROPIC_API_KEY set by the outer
    # wrapper to the LiteLLM key, so use that. Allow explicit override
    # via NEMO_RETRIEVER_JUDGE_API_KEY.
    api_key = (
        os.environ.get("NEMO_RETRIEVER_JUDGE_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "no judge API key set (NEMO_RETRIEVER_JUDGE_API_KEY / "
            "ANTHROPIC_API_KEY / NVIDIA_API_KEY)"
        )

    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get(
            "NEMO_RETRIEVER_JUDGE_BASE", "https://inference-api.nvidia.com/v1"
        ),
    )

    # LiteLLM proxies to Bedrock don't always honour response_format
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

    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        import re

        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if not match:
            raise ValueError(f"judge reply is not JSON: {raw_content[:200]!r}")
        return json.loads(match.group(0))


def to_check_rows(behaviors: list[str], grade: dict[str, Any]) -> list[dict[str, Any]]:
    """Pair grade["checks"] items with behaviour strings by position.

    If the grader returned fewer items than behaviours, pad with fails so
    the reward reflects the miss. Extra items are truncated.
    """
    grade_items = grade.get("checks") or []
    rows: list[dict[str, Any]] = []
    for i, behavior in enumerate(behaviors):
        item = grade_items[i] if i < len(grade_items) else {}
        rows.append(
            {
                "check": behavior,
                "pass": bool(item.get("pass", False)),
                "route": "agent",
                "matched": str(item.get("matched", "")),
                "rationale": str(item.get("rationale", "")),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--reward-file", default="/logs/verifier/reward.txt")
    parser.add_argument("--details-file", default="/logs/verifier/judge.json")
    args = parser.parse_args()

    spec = json.loads(Path(args.spec).read_text())
    expects = spec.get("expects") or []
    if args.step < 1 or args.step > len(expects):
        raise SystemExit(f"step {args.step} out of range for {len(expects)} expects")

    step = expects[args.step - 1]
    behaviors = step.get("expected_behavior") or []
    query = step.get("query", "")
    if not behaviors:
        raise SystemExit(
            f"{args.spec} expects[{args.step}] has no expected_behavior list"
        )

    judge_cfg = spec.get("judge") or {}
    model = judge_cfg.get("model") or DEFAULT_MODEL

    trajectory = read_trajectory()
    traj_path = locate_trajectory()
    prompt = build_prompt(query, behaviors, trajectory)

    try:
        grade = call_judge(prompt, model)
    except Exception as exc:  # noqa: BLE001 -- report cleanly, don't crash harbor
        print(f"JUDGE_ERROR: {exc}", file=sys.stderr)
        grade = {
            "checks": [
                {"pass": False, "matched": "", "rationale": f"judge failed: {exc}"}
                for _ in behaviors
            ]
        }

    results = to_check_rows(behaviors, grade)
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    reward = (passed / total) if total else 0.0

    log_dir = Path(args.reward_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    Path(args.reward_file).write_text(f"{reward}\n")
    Path(args.details_file).write_text(
        json.dumps(
            {
                "spec": args.spec,
                "step": args.step,
                "query": query,
                "total": total,
                "passed": passed,
                "reward": reward,
                "trajectory_path": str(traj_path) if traj_path else None,
                "trajectory_found": traj_path is not None,
                "checks": results,
            },
            indent=2,
        )
        + "\n"
    )

    for r in results:
        status = "PASS" if r["pass"] else "FAIL"
        print(f"{status}: {r['check']}")
        if r.get("rationale"):
            print(f"  {r['rationale']}")
    print(
        f"=== Results: {passed} passed, {total - passed} failed (of {total}); "
        f"reward={reward:.3f} ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
