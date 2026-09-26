"""Build the compact paper result layer from detailed evaluation outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DOMAINS = ("CUB", "EuroSAT", "ISIC")
KEY_MODELS = ("ProtoNet+ViT", "SupportTransformer+ViT", "WIPT-2")

TRAINED_VARIANTS = ("WIPT-4", "WIPT-6", "WIPT-Residual", "WIPT-All")

DESCRIPTIONS = {
    "paper/in_domain_accuracy.csv": "miniImageNet held-out accuracy across five evaluation seeds",
    "paper/cross_domain_accuracy.csv": "cross-domain accuracy and geometry summaries",
    "paper/cross_domain_paired_tests.csv": "paired WIPT comparisons on matched episodes",
    "paper/corruption_abc.csv": "custom A/B/C corruption results with frozen ProtoNet",
    "paper/eurosat_query_transitions.csv": "EuroSAT ProtoNet-to-WIPT decision transitions",
    "paper/eurosat_margin_bins.csv": "EuroSAT WIPT effect versus ProtoNet true-class margin",
    "paper/support_noise_summary.csv": "support-noise mechanism values for the support-noise diagnostic",
    "paper/design_ablations.csv": "combined WIPT design ablations",
    "figure_inputs/margin_geometry_paired.csv": "paired episode-level margins for the margin-geometry diagnostic",
}


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"required raw result not found: {path}")
    return path


def build_geometry(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in DOMAINS:
        data = episodes[(episodes.dataset == domain) & episodes.model.isin(KEY_MODELS)]
        wide = data.pivot(index="episode", columns="model", values="margin_norm")
        missing = set(KEY_MODELS) - set(wide.columns)
        if missing:
            raise ValueError(f"{domain}: missing margin data for {sorted(missing)}")
        current = pd.DataFrame(
            {
                "dataset": domain,
                "episode": wide.index.astype(int),
                "protonet_margin": wide["ProtoNet+ViT"].to_numpy(float),
                "support_transformer_margin": wide["SupportTransformer+ViT"].to_numpy(
                    float
                ),
                "wipt_margin": wide["WIPT-2"].to_numpy(float),
            }
        )
        current["wipt_minus_protonet"] = current.wipt_margin - current.protonet_margin
        current["wipt_minus_support_transformer"] = (
            current.wipt_margin - current.support_transformer_margin
        )
        rows.append(current)
    return pd.concat(rows, ignore_index=True)


def build_design_ablations(
    cross_domain: pd.DataFrame,
    inference: pd.DataFrame,
    margin: pd.DataFrame,
) -> pd.DataFrame:
    baseline = (
        cross_domain[cross_domain.model == "WIPT-2"]
        .set_index("dataset")["accuracy"]
        .to_dict()
    )
    rows = []

    for _, row in cross_domain[cross_domain.model.isin(TRAINED_VARIANTS)].iterrows():
        base = float(baseline[row.dataset])
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.model,
                "accuracy": float(row.accuracy),
                "baseline_accuracy": base,
                "delta_pp": float(row.accuracy) - base,
                "family": "trained_variant",
            }
        )

    for _, row in inference[inference.variant != "mean_euclid"].iterrows():
        base = float(baseline[row.dataset])
        family = (
            "query_weighted"
            if str(row.variant).startswith("query_weighted")
            else "inference_variant"
        )
        rows.append(
            {
                "dataset": row.dataset,
                "model": row.variant,
                "accuracy": float(row.accuracy),
                "baseline_accuracy": base,
                "delta_pp": float(row.accuracy) - base,
                "family": family,
            }
        )

    for _, row in margin.iterrows():
        base = float(baseline[row.dataset])
        rows.append(
            {
                "dataset": row.dataset,
                "model": "margin_objective",
                "accuracy": float(row.margin_accuracy),
                "baseline_accuracy": base,
                "delta_pp": float(row.margin_accuracy) - base,
                "family": "trained_variant",
            }
        )

    order = {name: index for index, name in enumerate(DOMAINS)}
    output = pd.DataFrame(rows)
    output["_domain_order"] = output.dataset.map(order)
    return (
        output.sort_values(["_domain_order", "family", "model"])
        .drop(columns="_domain_order")
        .reset_index(drop=True)
    )


def write_manifest(results: Path) -> None:
    rows = []
    for relative, description in DESCRIPTIONS.items():
        frame = pd.read_csv(require(results / relative))
        rows.append(
            {
                "file": relative,
                "rows": len(frame),
                "columns": ", ".join(frame.columns),
                "description": description,
            }
        )
    pd.DataFrame(rows).to_csv(results / "manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("results/raw"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()

    raw = args.raw
    results = args.results
    paper = results / "paper"
    figure_inputs = results / "figure_inputs"
    paper.mkdir(parents=True, exist_ok=True)
    figure_inputs.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(require(raw / "clean" / "in_domain_accuracy.csv"))
    cross = pd.read_csv(require(raw / "cross_domain" / "summary.csv"))
    tests = pd.read_csv(require(raw / "cross_domain" / "paired_tests.csv"))
    episodes = pd.read_csv(require(raw / "cross_domain" / "episodes.csv"))
    corruption = pd.read_csv(require(raw / "corruption" / "abc_corruption.csv"))
    transitions = pd.read_csv(
        require(raw / "query_analysis" / "eurosat_transitions.csv")
    )
    margin_bins = pd.read_csv(
        require(raw / "query_analysis" / "eurosat_margin_bins.csv")
    )
    noise = pd.read_csv(require(raw / "noise" / "support_noise_summary.csv"))
    noise.loc[noise.severity == 0, "noise_amplification_ratio"] = "not_applicable"
    inference = pd.read_csv(require(raw / "ablations" / "inference_variants.csv"))
    margin = pd.read_csv(require(raw / "ablations" / "margin_variant.csv"))

    clean.to_csv(paper / "in_domain_accuracy.csv", index=False)
    cross.to_csv(paper / "cross_domain_accuracy.csv", index=False)
    tests.to_csv(paper / "cross_domain_paired_tests.csv", index=False)
    corruption.to_csv(paper / "corruption_abc.csv", index=False)
    transitions.to_csv(paper / "eurosat_query_transitions.csv", index=False)
    margin_bins.to_csv(paper / "eurosat_margin_bins.csv", index=False)
    noise.to_csv(paper / "support_noise_summary.csv", index=False)
    build_design_ablations(cross, inference, margin).to_csv(
        paper / "design_ablations.csv", index=False
    )
    build_geometry(episodes).to_csv(
        figure_inputs / "margin_geometry_paired.csv", index=False
    )
    write_manifest(results)
    print(f"built compact paper results in {results}")


if __name__ == "__main__":
    main()
