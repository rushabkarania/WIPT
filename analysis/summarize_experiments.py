"""Collect completed experiment summaries into one report."""

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

EXPECTED = {
    "multiquery_1shot": "results/raw/multiquery/1shot/clean/query_group_summary.csv",
    "multiquery_5shot": "results/raw/multiquery/5shot/clean/query_group_summary.csv",
    "factorial_5shot": "results/raw/factorial/5shot/clean/condition_summary.csv",
    "factorial_comparisons_5shot": "results/raw/factorial/5shot/clean/adaptation_comparisons.csv",
    "meta_shift_1shot": "results/raw/meta_shift/1shot/evaluation/model_summary.csv",
    "meta_shift_5shot": "results/raw/meta_shift/5shot/evaluation/model_summary.csv",
    "mechanism_1shot": "results/raw/shot_domain_mechanism/1shot/domain_summary.csv",
    "mechanism_5shot": "results/raw/shot_domain_mechanism/5shot/domain_summary.csv",
    "efficiency_1shot": "results/raw/efficiency_1shot.csv",
    "efficiency_5shot": "results/raw/efficiency_5shot.csv",
}
for mode in ["shared", "cross", "mixed"]:
    for shot in [1, 5]:
        EXPECTED[f"shift_{mode}_{shot}shot"] = (
            f"results/raw/shift_study/{mode}/{shot}shot/clean/training_seed_summary.csv"
        )
        EXPECTED[f"shift_diff_{mode}_{shot}shot"] = (
            f"results/raw/shift_study/{mode}/{shot}shot/clean/difference_summary.csv"
        )


def markdown_table(frame: pd.DataFrame, max_rows: int = 100) -> str:
    frame = frame.head(max_rows).copy()
    if frame.empty:
        return "_(empty)_\n"
    # Avoid optional tabulate dependency.
    cols = list(frame.columns)

    def clean(v):
        if pd.isna(v):
            return ""
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(clean(row[c]) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--out-dir", type=Path, default=Path("results/summary"))
    args = p.parse_args()
    out = args.root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    status = []
    sections = ["# WIPT experiment summaries\n"]
    combined = []
    for name, rel in EXPECTED.items():
        path = args.root / rel
        exists = path.exists()
        status.append({"experiment": name, "path": rel, "complete": exists})
        if not exists:
            continue
        frame = pd.read_csv(path)
        tagged = frame.copy()
        tagged.insert(0, "experiment", name)
        combined.append(tagged)
        sections.append(f"\n## {name}\n\nSource: `{rel}`\n\n")
        sections.append(markdown_table(frame))

    status_frame = pd.DataFrame(status)
    status_frame.to_csv(out / "experiment_status.csv", index=False)
    if combined:
        # Columns differ across experiments, so concat produces a wide archival table.
        pd.concat(combined, ignore_index=True, sort=False).to_csv(
            out / "all_summary_rows.csv", index=False
        )
    sections.insert(
        1, "\n## Completion status\n\n" + markdown_table(status_frame) + "\n"
    )
    (out / "summary.md").write_text("".join(sections), encoding="utf-8")
    print(status_frame.to_string(index=False))
    print(f"wrote {out / 'summary.md'}")


if __name__ == "__main__":
    main()
