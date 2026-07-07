#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate Harbor tasks for the nemo-retriever skill.

Modeled on the AIQ-Sarath deterministic adapter pattern
(`sarathm/aiq-skill-eval-ci` on NVIDIA-AI-Blueprints/aiq), stripped of
AIQ-server assumptions since nemo-retriever is a CLI-only skill.

Two verifier modes:
    * deterministic (default) -- retriever_checks.py, hermetic, no LLM.
    * llm_judge               -- retriever_judge.py, grades trajectory
                                 against `expected_behavior` bullets via
                                 inference-api.nvidia.com. Opt in with
                                 `"mode": "llm_judge"` at spec top-level.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).resolve().parents[4]
VERIFIERS_DIR = Path(__file__).resolve().parents[2] / "verifiers"
VERIFIER = VERIFIERS_DIR / "retriever_checks.py"
JUDGE_VERIFIER = VERIFIERS_DIR / "retriever_judge.py"
PREAMBLE = (
    "You are running inside a non-interactive evaluation harness. "
    "Use the installed `/nemo-retriever` Agent Skill autonomously. Do not ask "
    "the user to run commands that you can run yourself."
)

PLATFORMS = {
    "local": {
        "short_name": "local",
        "description": "Runner-local or developer-workstation execution (CPU-only OK)",
    }
}


def _copytree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _render(value: str, *, platform: str, mode: str, repo_root: Path) -> str:
    replacements = {
        "{{platform}}": platform,
        "{{mode}}": mode,
        "{{repo_root}}": str(repo_root),
    }
    rendered = value
    for key, replacement in replacements.items():
        rendered = rendered.replace(key, replacement)
    return Template(rendered).safe_substitute(
        platform=platform,
        mode=mode,
        repo_root=str(repo_root),
    )


def _test_script(spec_name: str, step: int, verifier_script: str) -> str:
    return (
        "#!/bin/bash\n"
        "set -uo pipefail\n"
        'TEST_DIR="$(cd "$(dirname "$0")" && pwd)"\n'
        'LOCAL_SKILLS_DIR="$(cd "$TEST_DIR/.." && pwd)/skills"\n'
        'if [ -d "$LOCAL_SKILLS_DIR" ]; then\n'
        '  export NEMO_RETRIEVER_EVAL_SKILLS_DIR="${NEMO_RETRIEVER_EVAL_SKILLS_DIR:-$LOCAL_SKILLS_DIR}"\n'
        "fi\n"
        f'python3 "$TEST_DIR/{verifier_script}" --spec "$TEST_DIR/{spec_name}" --step {step}\n'
        "exit 0\n"
    )


def _solution_script() -> str:
    return (
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "# Trivial health probe: the skill install doc guarantees the CLI on PATH.\n"
        'command -v retriever && retriever --help | head -5\n'
    )


def _environment_compose(judge_mode: bool = False) -> str:
    # Passes NVIDIA_API_KEY through from the host so hosted embed NIMs on
    # integrate.api.nvidia.com are reachable. In llm_judge mode also pass
    # NEMO_RETRIEVER_JUDGE_API_KEY (LiteLLM sk-... key) so the judge can
    # hit inference-api.nvidia.com from the verifier container.
    judge_env_line = (
        "      NEMO_RETRIEVER_JUDGE_API_KEY: ${NEMO_RETRIEVER_JUDGE_API_KEY}\n"
        if judge_mode
        else ""
    )
    return (
        "services:\n"
        "  main:\n"
        "    extra_hosts:\n"
        '      - "host.docker.internal:host-gateway"\n'
        "    environment:\n"
        "      NVIDIA_API_KEY: ${NVIDIA_API_KEY}\n"
        f"{judge_env_line}"
        "    volumes:\n"
        "      - type: bind\n"
        "        source: ${CONTEXT_DIR}/../skills\n"
        "        target: /skills\n"
        "        read_only: true\n"
    )


def _environment_dockerfile(judge_mode: bool = False) -> str:
    # Slim image + retriever library. Kept intentionally minimal: pip
    # install is done at container-build time, not at task-run time. In
    # llm_judge mode, also install `openai` so retriever_judge.py can
    # call inference-api.nvidia.com.
    extra_pip = " openai" if judge_mode else ""
    return (
        "FROM python:3.12-slim\n"
        "RUN apt-get update && apt-get install -y --no-install-recommends \\\n"
        "        curl ca-certificates \\\n"
        "    && rm -rf /var/lib/apt/lists/*\n"
        f"RUN pip install --no-cache-dir nemo-retriever{extra_pip}\n"
        'ENV RETRIEVER_VENV=/usr/local\n'
    )


def _task_toml(
    skill: str,
    spec_stem: str,
    platform: str,
    mode: str,
    step: int,
    total_steps: int,
) -> str:
    platform_short = PLATFORMS[platform]["short_name"]
    step_suffix = f"-step-{step}" if total_steps > 1 else ""
    return "\n".join(
        [
            "[task]",
            f'name = "nvidia-nemo-retriever/{skill}-{spec_stem}-{platform_short}-{mode}{step_suffix}"',
            f'description = "{skill} {spec_stem} eval on {platform}/{mode} step {step} of {total_steps}"',
            f'keywords = ["{skill}", "{spec_stem}", "{platform}", "{mode}"]',
            "",
            "[environment]",
            'skills_dir = "/skills"',
            "",
            "[metadata]",
            f'skill = "{skill}"',
            f'spec = "{spec_stem}"',
            f'platform = "{platform}"',
            f'mode = "{mode}"',
            f"step = {step}",
            f"total_steps = {total_steps}",
            "",
        ]
    )


def generate(spec_path: Path, skill_dir: Path, output_dir: Path, repo_root: Path) -> list[Path]:
    spec = json.loads(spec_path.read_text())
    skill = skill_dir.name
    spec_stem = spec_path.stem
    expects = spec.get("expects") or []
    if not expects:
        raise ValueError(f"{spec_path} has no expects entries")

    judge_mode = spec.get("mode", "deterministic") == "llm_judge"
    verifier_src = JUDGE_VERIFIER if judge_mode else VERIFIER
    verifier_filename = verifier_src.name

    generated: list[Path] = []
    platforms = spec.get("resources", {}).get("platforms", {})
    for platform, platform_cfg in platforms.items():
        if platform not in PLATFORMS:
            raise ValueError(f"unsupported platform {platform!r}; supported: {sorted(PLATFORMS)}")
        for mode in platform_cfg.get("modes", []):
            mode_slug = mode.lower().replace("_", "-")
            base_dir = output_dir / spec_stem / f"{PLATFORMS[platform]['short_name']}-{mode_slug}"
            for idx, expect in enumerate(expects, start=1):
                step_dir = base_dir / f"step-{idx}" if len(expects) > 1 else base_dir
                tests_dir = step_dir / "tests"
                solution_dir = step_dir / "solution"
                skills_dir = step_dir / "skills"
                env_dir = step_dir / "environment"
                tests_dir.mkdir(parents=True, exist_ok=True)
                solution_dir.mkdir(parents=True, exist_ok=True)
                skills_dir.mkdir(parents=True, exist_ok=True)
                env_dir.mkdir(parents=True, exist_ok=True)

                instruction = "\n".join(
                    [
                        PREAMBLE,
                        "",
                        "Use the `/nemo-retriever` skill for this task.",
                        "",
                        f"## Query {idx} of {len(expects)}",
                        "",
                        _render(expect.get("query", ""), platform=platform, mode=mode, repo_root=repo_root),
                        "",
                        "## Environment Notes",
                        "",
                        _render(spec.get("env", ""), platform=platform, mode=mode, repo_root=repo_root),
                        "",
                    ]
                )
                (step_dir / "instruction.md").write_text(instruction + "\n")
                (step_dir / "task.toml").write_text(
                    _task_toml(skill, spec_stem, platform, mode_slug, idx, len(expects))
                )
                (env_dir / "Dockerfile").write_text(_environment_dockerfile(judge_mode=judge_mode))
                (env_dir / "docker-compose.yaml").write_text(_environment_compose(judge_mode=judge_mode))
                (tests_dir / spec_path.name).write_text(json.dumps(spec, indent=2) + "\n")
                shutil.copy2(verifier_src, tests_dir / verifier_filename)
                test_sh = tests_dir / "test.sh"
                test_sh.write_text(_test_script(spec_path.name, idx, verifier_filename))
                test_sh.chmod(0o755)
                solve_sh = solution_dir / "solve.sh"
                solve_sh.write_text(_solution_script())
                solve_sh.chmod(0o755)

                _copytree(skill_dir, skills_dir / skill)
            generated.append(base_dir)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skill-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    args = parser.parse_args()

    generated = generate(
        spec_path=Path(args.spec).resolve(),
        skill_dir=Path(args.skill_dir).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        repo_root=Path(args.repo_root).resolve(),
    )
    print(json.dumps({"generated": [str(path) for path in generated]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
