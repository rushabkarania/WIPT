"""Validate the compact result snapshots committed with the release."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PAPER = RESULTS / "paper"
DOMAINS = {"CUB", "EuroSAT", "ISIC"}
KEY_MODELS = {"ProtoNet+ViT", "SupportTransformer+ViT", "WIPT-2"}
EXPECTED_FROZEN_PROTONET = {
    "CUB": 97.880005,
    "EuroSAT": 82.7626662,
    "ISIC": 38.9182224,
}
EXPECTED_TRANSITIONS = {
    "Wrong->Wrong": 17287,
    "Wrong->Correct": 2105,
    "Correct->Wrong": 2599,
    "Correct->Correct": 90509,
}


def main() -> None:
    clean = pd.read_csv(PAPER / "in_domain_accuracy.csv")
    clean_proto = float(clean.loc[clean.model == "ProtoNet+ViT", "mean"].iloc[0])
    assert np.isclose(clean_proto, 98.3369, atol=1e-4)

    cross = pd.read_csv(PAPER / "cross_domain_accuracy.csv")
    assert set(cross.dataset) == DOMAINS
    proto = cross[cross.model == "ProtoNet+ViT"].set_index("dataset")
    assert len(proto) == 3
    for domain, expected in EXPECTED_FROZEN_PROTONET.items():
        assert np.isclose(float(proto.loc[domain, "accuracy"]), expected, atol=1e-5)

    tests = pd.read_csv(PAPER / "cross_domain_paired_tests.csv")
    assert len(tests) == 6
    assert set(tests.dataset) == DOMAINS
    assert (tests.n_episodes == 1500).all()

    corruption = pd.read_csv(PAPER / "corruption_abc.csv")
    assert len(corruption) == 144
    assert set(corruption.dataset) == DOMAINS
    assert set(corruption.model) == KEY_MODELS
    for domain in DOMAINS:
        assert len(corruption[corruption.dataset == domain]) == 48

    transitions = pd.read_csv(PAPER / "eurosat_query_transitions.csv")
    counts = dict(
        zip(
            transitions.protonet_state + "->" + transitions.wipt_state,
            transitions.n_queries,
        )
    )
    assert counts == EXPECTED_TRANSITIONS

    bins = pd.read_csv(PAPER / "eurosat_margin_bins.csv")
    assert len(bins) == 20
    assert int(bins.n_queries.sum()) == 112500

    noise = pd.read_csv(PAPER / "support_noise_summary.csv")
    assert set(noise.severity) == {0, 3, 5}
    assert set(noise.model) == {"ProtoNet", "WIPT"}
    assert len(noise) == 6
    assert (
        noise.loc[noise.severity == 0, "noise_amplification_ratio"] == "not_applicable"
    ).all()
    pd.to_numeric(
        noise.loc[noise.severity > 0, "noise_amplification_ratio"], errors="raise"
    )

    ablations = pd.read_csv(PAPER / "design_ablations.csv")
    assert len(ablations) == 33
    assert set(ablations.dataset) == DOMAINS

    geometry = pd.read_csv(RESULTS / "figure_inputs" / "margin_geometry_paired.csv")
    assert len(geometry) == 4500
    assert set(geometry.dataset) == DOMAINS
    for domain in DOMAINS:
        domain_rows = geometry[geometry.dataset == domain]
        assert len(domain_rows) == 1500
        assert domain_rows.episode.nunique() == 1500

    manifest = pd.read_csv(RESULTS / "manifest.csv")
    assert len(manifest) == 9

    print("result validation passed")


if __name__ == "__main__":
    main()
