"""Run lightweight checks for the repository without model weights."""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_python_syntax() -> None:
    failures = []
    for path in sorted(ROOT.rglob("*.py")):
        if set(path.parts) & {"__pycache__", ".venv", "venv", "env"}:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")
    if failures:
        raise RuntimeError("Python syntax errors:\n" + "\n".join(failures))


def run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures", action="store_true", help="regenerate the PNG figures"
    )
    args = parser.parse_args()

    check_python_syntax()
    run_module("analysis.validate_results")
    if args.figures:
        run_module("scripts.generate_figures")
    print("repository checks passed")


if __name__ == "__main__":
    main()
