"""Generate the manuscript figures from retained CSV results.

Run python -m scripts.generate_figures from the repository root.
No datasets, checkpoints or model dependencies are required.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

DOMAINS = ["CUB", "EuroSAT", "ISIC"]
BLUE, RED, GOLD, GRAY = "#0072B2", "#D55E00", "#B28200", "#525A66"
COLORS = dict(zip(DOMAINS, [BLUE, RED, "#009E73"]))
FIGURE_NAMES = [
    "architecture",
    "cross_domain_performance",
    "multiquery_efficiency",
    "shot_domain_mechanism",
    "boundary_analysis",
    "corruption_response",
    "margin_geometry",
    "support_noise",
]


def required_inputs():
    paths = []
    for shot in (1, 5):
        paths += [
            f"raw/training_seed_study/{shot}shot/clean/{name}.csv"
            for name in ("run_accuracy", "difference_summary")
        ]
        paths += [
            f"raw/multiquery/{shot}shot/clean/difference_vs_q1_summary.csv",
            f"raw/efficiency_{shot}shot.csv",
            f"raw/shot_domain_mechanism/{shot}shot/domain_summary.csv",
        ]
        paths += [
            f"raw/boundary_all_domains/{shot}shot/{d}/queries_across_training_seeds.csv"
            for d in DOMAINS
        ]
    return paths + [
        "raw/training_seed_study/5shot/corruption/corruption_gap_by_training_seed.csv",
        "figure_inputs/margin_geometry_paired.csv",
        "paper/support_noise_summary.csv",
    ]


def locate_input(root, relative):
    candidates = [root / relative]
    if relative.startswith("raw/"):
        candidates.append(root / relative[4:])
    if root.name == "raw":
        candidates.append(root.parent / relative)
    return next((p for p in candidates if p.is_file()), None)


def resolve_results(override):
    script_dir = Path(__file__).resolve().parent
    if override is not None:
        candidates = [override.expanduser().resolve()]
    else:
        candidates = []
        for base in (Path.cwd(), script_dir, script_dir.parent):
            candidates += [
                base / "results",
                base / "data/results",
                base / "final_results/results",
            ]
    for root in candidates:
        if locate_input(root, "raw/training_seed_study/1shot/clean/run_accuracy.csv"):
            return root.resolve()
    raise FileNotFoundError(
        "Could not locate the results folder. Use --results PATH.\nSearched:\n  "
        + "\n  ".join(map(str, candidates))
    )


def stats(a):
    a = np.asarray(a, dtype=float)
    return (
        a.mean(),
        a.std(ddof=1),
        t.ppf(0.975, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)),
    )


def zero(ax):
    ax.axvline(0, color="#78828E", lw=0.8, ls="--", zorder=0)
    ax.grid(axis="x", alpha=0.15, zorder=0)


def draw_architecture(save):
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, ax = plt.subplots(figsize=(7.15, 6.15))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    ax.set(xlim=(0, 15.8), ylim=(0, 14))
    ax.axis("off")
    SUP = "#F7E3D1"
    QUE = "#DDEAF5"
    TR = "#D8EDE3"
    PRO = "#EBE1F2"
    GRAY = "#F2F3F4"
    SC = "#B76A30"
    QC = "#376F9B"
    TC = "#37815D"
    PC = "#80579B"

    def box(cx, cy, w, h, label, fill, fs=7.4, bold=False):
        ax.add_patch(
            FancyBboxPatch(
                (cx - w / 2, cy - h / 2),
                w,
                h,
                boxstyle="round,pad=.025,rounding_size=.06",
                facecolor=fill,
                edgecolor="#5C6268",
                linewidth=0.7,
                zorder=2,
            )
        )
        ax.text(
            cx,
            cy,
            label,
            ha="center",
            va="center",
            fontsize=fs,
            weight="bold" if bold else "normal",
            linespacing=1.12,
            zorder=3,
        )

    def arr(x1, y1, x2, y2, c):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "-|>",
                "lw": 0.9,
                "color": c,
                "shrinkA": 2,
                "shrinkB": 2,
            },
            zorder=1,
        )

    ax.text(
        7.9,
        13.7,
        "Shared frozen ViT-S/16 checkpoint • 384-dimensional image embeddings",
        ha="center",
        va="center",
        fontsize=8.1,
    )
    for j, (title, subtitle) in enumerate(
        [
            ("(a) ProtoNet", "Direct support averaging"),
            ("(b) Support-only Transformer", "Support transformation"),
            ("(c) WIPT", "Joint query–support transformation"),
        ]
    ):
        off = j * 5.3
        left = off + 1.25
        right = off + 4.0
        mid = off + 2.625
        ax.text(mid, 13.02, title, ha="center", fontsize=8.1, weight="bold")
        ax.text(mid, 12.57, subtitle, ha="center", fontsize=6.7, color="#525960")
        box(left, 11.78, 2.12, 0.8, "Support\nimages", SUP)
        box(right, 11.78, 2.12, 0.8, "Query\nimage", QUE)
        box(mid, 10.34, 4.45, 0.88, "Frozen ViT-S/16\n(shared checkpoint)", GRAY)
        arr(left, 11.38, mid - 0.65, 10.78, SC)
        arr(right, 11.38, mid + 0.65, 10.78, QC)
        box(left, 8.92, 2.12, 0.8, "Support\nembeddings", SUP)
        box(right, 8.92, 2.12, 0.8, "Query\nembedding", QUE)
        arr(mid - 0.65, 9.9, left, 9.32, SC)
        arr(mid + 0.65, 9.9, right, 9.32, QC)
        if j == 0:
            arr(left, 8.52, left, 4.73, SC)
            arr(right, 8.52, mid + 0.8, 1.35, QC)
        elif j == 1:
            box(
                left,
                7.37,
                2.12,
                1.05,
                "Support-only\nTransformer\n2 layers",
                TR,
                fs=7.0,
            )
            arr(left, 8.52, left, 7.895, SC)
            box(left, 5.82, 2.12, 0.84, "Transformed\nsupports", TR)
            arr(left, 6.845, left, 6.24, TC)
            arr(left, 5.4, left, 4.73, TC)
            arr(right, 8.52, mid + 0.8, 1.35, QC)
        else:
            box(
                mid,
                7.37,
                4.45,
                1.05,
                "Joint self-attention\nquery + support embeddings\n2-layer Transformer",
                TR,
                fs=7.5,
            )
            arr(left, 8.52, mid - 0.8, 7.895, SC)
            arr(right, 8.52, mid + 0.8, 7.895, QC)
            box(left, 5.82, 2.12, 0.84, "Transformed\nsupports", TR)
            box(right, 5.82, 2.12, 0.84, "Transformed\nquery", QUE)
            arr(mid - 0.8, 6.845, left, 6.24, TC)
            arr(mid + 0.8, 6.845, right, 6.24, QC)
            arr(left, 5.4, left, 4.73, TC)
            arr(right, 5.4, mid + 0.8, 1.35, QC)
        box(left, 4.28, 2.12, 0.9, "Class-wise\nmean", PRO)
        box(
            left,
            2.8,
            2.12,
            0.88,
            "Query-specific\nprototypes" if j == 2 else "Class\nprototypes",
            PRO,
            fs=7.1,
        )
        arr(left, 3.83, left, 3.24, PC)
        box(mid, 0.88, 3.6, 0.94, "Negative Euclidean\ndistance", GRAY, fs=7.7)
        arr(left, 2.36, mid - 0.65, 1.35, PC)
    ax.text(
        7.9,
        0.08,
        "Support labels define class means. Query labels are not used for prediction.",
        ha="center",
        fontsize=7,
    )
    save(fig, "architecture", pad=0.05)


def generate(results, out, dpi=300, pdf=False):
    missing = [p for p in required_inputs() if locate_input(results, p) is None]
    if missing:
        raise FileNotFoundError(
            "Required CSV files are missing under "
            + str(results)
            + ":\n  "
            + "\n  ".join(missing)
            + "\nUse the complete results export; the six query-level files are needed for the boundary plot."
        )
    out.mkdir(parents=True, exist_ok=True)
    DERIVED = out / "derived"
    DERIVED.mkdir(exist_ok=True)
    generated = []

    def read(relative):
        return pd.read_csv(locate_input(results, relative))

    def save(fig, name, pad=0.08):
        path = out / (name + ".png")
        fig.savefig(
            path, dpi=dpi, bbox_inches="tight", pad_inches=pad, facecolor="white"
        )
        if pdf:
            fig.savefig(
                out / (name + ".pdf"),
                bbox_inches="tight",
                pad_inches=pad,
                facecolor="white",
            )
        plt.close(fig)
        generated.append(name)
        print("Saved:", path, flush=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#88909B",
            "axes.linewidth": 0.6,
            "savefig.dpi": 180,
        }
    )

    # Recalculate primary comparisons from the run-level results.
    primary = []
    primary_runs = {}
    for shot in [1, 5]:
        runs = read(f"raw/training_seed_study/{shot}shot/clean/run_accuracy.csv")
        summary = read(
            f"raw/training_seed_study/{shot}shot/clean/difference_summary.csv"
        )
        for d in DOMAINS:
            z = runs[runs.dataset.eq(d)]
            pn = float(z[z.model.eq("ProtoNet")].accuracy.iloc[0])
            w = z[z.model.eq("WIPT")].sort_values("training_seed").accuracy.to_numpy()
            so = (
                z[z.model.eq("Support-only Transformer")]
                .sort_values("training_seed")
                .accuracy.to_numpy()
            )
            assert len(w) == len(so) == 5
            primary_runs[shot, d] = (pn, so, w)
            for model, comp in [("ProtoNet", pn), ("Support-only Transformer", so)]:
                diff = w - comp
                mean, sd, ci = stats(diff)
                orig = summary[
                    (summary.dataset.eq(d)) & summary.comparison.eq("WIPT - " + model)
                ].iloc[0]
                assert np.isclose(mean, orig.mean_difference_pp, atol=1e-9)
                assert np.isclose(ci, orig.training_seed_ci95_pp, atol=1e-5)
                primary.append(
                    dict(
                        shot=shot,
                        dataset=d,
                        comparator=model,
                        mean_pp=mean,
                        sd_pp=sd,
                        ci95_halfwidth_pp=ci,
                        lower_pp=mean - ci,
                        upper_pp=mean + ci,
                        positive_runs=int((diff > 0).sum()),
                        p_unadjusted=2 * t.sf(abs(mean / (sd / np.sqrt(5))), 4),
                    )
                )
    pd.DataFrame(primary).to_csv(DERIVED / "primary_paired_statistics.csv", index=False)

    # Recalculate rescue/break contributions and non-straddling boundary bins.
    transitions = []
    bins = []
    localization = []
    for shot in [1, 5]:
        for d in DOMAINS:
            q = read(
                f"raw/boundary_all_domains/{shot}shot/{d}/queries_across_training_seeds.csv"
            )
            n = len(q)
            assert n == 112500 and q.query_id.is_unique
            pn = q.protonet_correct.to_numpy().astype(bool)
            margin = q.protonet_margin.to_numpy()
            assert not (margin == 0).any()
            assert ((margin > 0) == pn).all()
            w = q[[f"wipt_correct_seed_{s}" for s in range(5)]].to_numpy().astype(bool)
            rescue = (~pn[:, None]) & w
            broken = pn[:, None] & (~w)
            net = (rescue.sum(0) - broken.sum(0)) / n * 100
            assert np.allclose(
                net, primary_runs[shot, d][2] - primary_runs[shot, d][0], atol=2e-5
            )
            for s in range(5):
                transitions.append(
                    dict(
                        shot=shot,
                        dataset=d,
                        training_seed=s,
                        n_queries=n,
                        pn_errors=int((~pn).sum()),
                        pn_successes=int(pn.sum()),
                        rescued=int(rescue[:, s].sum()),
                        broken=int(broken[:, s].sum()),
                        rescue_rate_percent=rescue[:, s].sum() / (~pn).sum() * 100,
                        break_rate_percent=broken[:, s].sum() / pn.sum() * 100,
                        rescued_per100=rescue[:, s].mean() * 100,
                        broken_per100=broken[:, s].mean() * 100,
                        net_pp=net[s],
                    )
                )
            for state in [False, True]:
                idx = np.flatnonzero(pn == state)
                idx = idx[np.argsort(margin[idx], kind="stable")]
                for bi, ids in enumerate(np.array_split(idx, 10), 1):
                    flip = (w[ids] != pn[ids, None]).mean(axis=0) * 100
                    mean, sd, ci = stats(flip)
                    bins.append(
                        dict(
                            shot=shot,
                            dataset=d,
                            pn_state="correct" if state else "wrong",
                            bin=bi,
                            n_queries=len(ids),
                            margin_min=margin[ids].min(),
                            margin_max=margin[ids].max(),
                            median_margin=np.median(margin[ids]),
                            flip_percent=mean,
                            sd=sd,
                            ci95=ci,
                            **{f"seed_{s}": flip[s] for s in range(5)},
                        )
                    )
            nearest = np.argsort(np.abs(margin), kind="stable")[: int(n * 0.2)]
            flips = w != pn[:, None]
            frac = flips[nearest].sum(0) / flips.sum(0) * 100
            localization.append(
                dict(
                    shot=shot,
                    dataset=d,
                    query_fraction_percent=20,
                    abs_margin_cutoff=float(np.abs(margin[nearest]).max()),
                    mean_fraction_of_flips_percent=frac.mean(),
                    minimum_over_seeds_percent=frac.min(),
                    maximum_over_seeds_percent=frac.max(),
                )
            )
    tr = pd.DataFrame(transitions)
    bn = pd.DataFrame(bins)
    tr.to_csv(DERIVED / "rescue_break_contributions_by_seed.csv", index=False)
    bn.to_csv(DERIVED / "boundary_bins_split_at_zero.csv", index=False)
    pd.DataFrame(localization).to_csv(
        DERIVED / "boundary_localization.csv", index=False
    )

    # Draw the three controlled representation pathways.
    draw_architecture(save)

    # Primary paired effects: each seed is visible; absolute accuracies are in Table 3.
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 3.65), sharex=True, sharey=True, layout="constrained"
    )
    for ax, shot in zip(axes, [1, 5]):
        for j, d in enumerate(DOMAINS):
            pn, so, w = primary_runs[shot, d]
            for k, (comp, color) in enumerate([(pn, BLUE), (so, RED)]):
                y = 5 - (j * 2 + k)
                diff = w - comp
                mean, sd, ci = stats(diff)
                ax.scatter(
                    diff,
                    y + np.linspace(-0.11, 0.11, 5),
                    s=14,
                    c=color,
                    alpha=0.45,
                    zorder=2,
                )
                ax.errorbar(
                    mean, y, xerr=ci, fmt="D", ms=4, c=color, capsize=3, zorder=3
                )
        zero(ax)
        ax.set_title(f"{shot}-shot", loc="left", weight="bold")
        ax.set_yticks(
            range(6),
            [
                "ISIC · SO",
                "ISIC · PN",
                "EuroSAT · SO",
                "EuroSAT · PN",
                "CUB · SO",
                "CUB · PN",
            ],
        )
        ax.set(
            xlabel="WIPT − comparator (percentage points)",
            ylim=(-0.6, 5.6),
            xlim=(-2.5, 3.1),
        )
        ax.axhline(1.5, c="#DFE3E8", lw=0.6)
        ax.axhline(3.5, c="#DFE3E8", lw=0.6)
    save(fig, "cross_domain_performance")

    # Multi-query accuracy and computational trade-offs, without dual y axes.
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), layout="constrained")
    for ax, shot in zip(axes[0], [1, 5]):
        z = read(f"raw/multiquery/{shot}shot/clean/difference_vs_q1_summary.csv")
        for i, d in enumerate(DOMAINS):
            a = z[z.dataset.eq(d)]
            ax.errorbar(
                a.query_group_size + (i - 1) * 0.065,
                a.mean_difference_vs_q1_pp,
                yerr=a.training_seed_ci95_pp,
                fmt="o-",
                ms=3,
                lw=1,
                c=COLORS[d],
                capsize=2,
                label=d,
            )
        ax.axhline(0, c=GRAY, lw=0.7, ls="--")
        ax.set_title(f"{shot}-shot accuracy", loc="left", weight="bold")
        ax.set(
            xlabel="Query group size, g",
            ylabel="Change from g = 1 (pp)",
            xticks=[1, 2, 3, 4, 5],
            xlim=(0.7, 5.3),
            ylim=(-1.65, 0.8),
        )
    axes[0, 0].legend(frameon=False, loc="lower left", ncol=1)
    eff = {s: read(f"raw/efficiency_{s}shot.csv").iloc[2:7].copy() for s in [1, 5]}
    for ax, col, title in [
        (axes[1, 0], "attention_token_pairs_per_episode", "Analytical attention pairs"),
        (axes[1, 1], "peak_memory_mb", "Recorded peak memory"),
    ]:
        for shot, color in [(1, BLUE), (5, RED)]:
            z = eff[shot]
            ax.plot(
                z.query_group_size,
                z[col] / z[col].iloc[0],
                "o-",
                c=color,
                label=f"{shot}-shot",
                ms=4,
            )
        ax.axhline(1, c=GRAY, lw=0.7, ls="--")
        ax.set_title(title, loc="left", weight="bold")
        ax.set(
            xlabel="Query group size, g",
            ylabel="Relative to g = 1",
            xticks=[1, 2, 3, 4, 5],
            ylim=(0, 1.2),
        )
    axes[1, 0].legend(frameon=False)
    save(fig, "multiquery_efficiency")

    # Contributions have the same denominator, so their sum is exactly the net gain.
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 5.0), layout="constrained")
    for ax, shot in zip(axes[0], [1, 5]):
        for i, d in enumerate(DOMAINS):
            z = tr[tr.shot.eq(shot) & tr.dataset.eq(d)]
            rr = z.rescued_per100.mean()
            bb = z.broken_per100.mean()
            m, sd, ci = stats(z.net_pp)
            ax.barh(
                2 - i, rr, color=BLUE, height=0.5, label="Rescued" if i == 0 else None
            )
            ax.barh(
                2 - i, -bb, color=RED, height=0.5, label="Broken" if i == 0 else None
            )
            ax.errorbar(
                m, 2 - i, xerr=ci, fmt="D", color="#151B24", ms=4, capsize=3, zorder=3
            )
        zero(ax)
        ax.set_title(
            f"{shot}-shot: actual accuracy contributions",
            loc="left",
            weight="bold",
            fontsize=9,
        )
        ax.set(
            yticks=[2, 1, 0],
            yticklabels=DOMAINS,
            xlim=(-6.1, 6.8),
            xlabel="Changed predictions per 100 queries",
            ylim=(-0.7, 2.7),
        )
        ax.set_xticks([-6, -3, 0, 3, 6])
    axes[0, 0].legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=6, label="Rescued"),
            Line2D([0], [0], color=RED, lw=6, label="Broken"),
            Line2D([0], [0], color="#151B24", marker="D", label="Net ± 95% CI", lw=0.8),
        ],
        loc="lower left",
        bbox_to_anchor=(-0.05, -0.46),
        frameon=False,
        ncol=3,
        columnspacing=0.7,
        handlelength=1.2,
        fontsize=7,
    )
    for ax, col, title in [
        (
            axes[1, 0],
            "support_to_query_centroid_error_norm",
            "Support-centroid error (oracle)",
        ),
        (
            axes[1, 1],
            "conditioning_variability_norm",
            "Query-conditioned prototype variability",
        ),
    ]:
        for j, shot in enumerate([1, 5]):
            z = read(f"raw/shot_domain_mechanism/{shot}shot/domain_summary.csv")
            a = z[z.metric.eq(col)].set_index("dataset").loc[DOMAINS]
            ax.errorbar(
                np.arange(3) + (j - 0.5) * 0.14,
                a["mean"],
                yerr=a.training_seed_ci95,
                fmt="o",
                capsize=3,
                c=[BLUE, RED][j],
                label=f"{shot}-shot",
            )
        ax.set_title(title, loc="left", weight="bold", fontsize=9)
        ax.set(
            xticks=range(3), xticklabels=DOMAINS, ylabel="Normalized diagnostic value"
        )
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.15)
    axes[1, 1].legend(frameon=False)
    save(fig, "shot_domain_mechanism")

    # Boundary flip probabilities: no bin crosses the correctness boundary.
    fig, axes = plt.subplots(
        2, 3, figsize=(6.8, 5.0), sharey=True, layout="constrained"
    )
    for row, shot in enumerate([1, 5]):
        for col, d in enumerate(DOMAINS):
            ax = axes[row, col]
            a = bn[bn.shot.eq(shot) & bn.dataset.eq(d)]
            for state, color in [("wrong", BLUE), ("correct", RED)]:
                z = a[a.pn_state.eq(state)]
                ax.errorbar(
                    z.median_margin,
                    z.flip_percent,
                    yerr=z.ci95,
                    fmt="o-",
                    ms=2.5,
                    lw=0.9,
                    c=color,
                    capsize=1.7,
                )
            zero(ax)
            ax.set_title(f"{d} · {shot}-shot", loc="left", weight="bold")
            ax.set(xlabel="Raw ProtoNet true-class margin", ylim=(-1, 56))
            if col == 0:
                ax.set_ylabel("Rescue / break within bin (%)")
            err = (
                tr[tr.shot.eq(shot) & tr.dataset.eq(d)].pn_errors.iloc[0] / 112500 * 100
            )
            ax.text(
                0.97,
                0.97,
                f"PN errors: {err:.2f}%",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
            )
    fig.legend(
        handles=[
            Line2D([0], [0], c=BLUE, marker="o", label="Rescue among PN errors"),
            Line2D([0], [0], c=RED, marker="o", label="Break among PN successes"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    save(fig, "boundary_analysis")

    # Corruption paired intervals, re-aggregated from seed-level differences.
    c = read(
        "raw/training_seed_study/5shot/corruption/corruption_gap_by_training_seed.csv"
    )
    fig, axes = plt.subplots(
        1, 2, figsize=(6.8, 3.1), sharey=True, layout="constrained"
    )
    for ax, comp, title in zip(
        axes,
        ["WIPT - ProtoNet", "WIPT - Support-only Transformer"],
        ["Relative to ProtoNet", "Relative to support-only"],
    ):
        for i, (cat, label) in enumerate(
            [
                ("A_structured", "A: brightness + contrast"),
                ("B_stochastic", "B: Gaussian noise"),
                ("C_combined", "C: combined"),
            ]
        ):
            a = c[c.comparison.eq(comp) & c.category.isin(["clean", cat])]
            vals = []
            for sev, z in a.groupby("severity"):
                mean, sd, ci = stats(z.difference_pp)
                vals.append((sev, mean, ci))
            vals = np.asarray(vals)
            ax.errorbar(
                vals[:, 0] + (i - 1) * 0.045,
                vals[:, 1],
                yerr=vals[:, 2],
                c=[BLUE, GOLD, RED][i],
                fmt="o-",
                ms=3,
                capsize=2,
                lw=1,
                label=label,
            )
        ax.axhline(0, c=GRAY, ls="--", lw=0.7)
        ax.set_title(title, loc="left", weight="bold")
        ax.set(
            xlabel="Corruption severity (0 = clean)",
            xticks=range(6),
            ylabel="WIPT − comparator (pp)",
        )
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    save(fig, "corruption_response")

    # Empirical cumulative distributions of paired episode-margin differences.
    g = read("figure_inputs/margin_geometry_paired.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.65), layout="constrained")
    for ax, d in zip(axes, DOMAINS):
        z = g[g.dataset.eq(d)]
        for comp, color in [("protonet", BLUE), ("support_transformer", RED)]:
            delta = z.wipt_margin - z[comp + "_margin"]
            # Empirical distribution rather than an inferential error bar.
            ys = np.sort(delta)
            ax.plot(
                ys,
                np.arange(1, len(ys) + 1) / len(ys),
                c=color,
                lw=1.2,
                label="PN" if comp == "protonet" else "SO",
            )
        zero(ax)
        ax.set_title(d, loc="left", weight="bold")
        ax.set(
            xlabel="WIPT − comparator margin",
            ylabel="Fraction of episodes",
            ylim=(0, 1),
        )
    axes[0].legend(frameon=False, title="Comparator", fontsize=7, title_fontsize=8)
    save(fig, "margin_geometry")

    # Selected-checkpoint diagnostic; no training-seed interval is available.
    n = read("paper/support_noise_summary.csv")
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.7), layout="constrained")
    for ax, col, title in zip(
        axes,
        ["noise_amplification_ratio", "absolute_margin", "accuracy"],
        ["Pooling attenuation", "Margin / clean margin", "Accuracy (%)"],
    ):
        for model, color in [("ProtoNet", GRAY), ("WIPT", BLUE)]:
            z = n[n.model.eq(model)].copy()
            vals = pd.to_numeric(z[col], errors="coerce")
            if col == "absolute_margin":
                vals = vals / vals.iloc[0]
            ax.plot(z.severity, vals, "o-", c=color, ms=4, label=model)
        ax.set_title(title, loc="left", weight="bold", fontsize=9)
        ax.set(xlabel="Support-noise severity", xticks=[0, 3, 5])
        ax.grid(axis="y", alpha=0.15)
    axes[0].legend(frameon=False)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 105)
    save(fig, "support_noise")

    if generated != FIGURE_NAMES:
        raise RuntimeError(
            "The generated figure list does not match the current manuscript."
        )
    summary = {
        "figure_count": len(generated),
        "format": "png",
        "dpi": dpi,
        "pdf_also_generated": pdf,
        "results": "results",
        "figures": [name + ".png" for name in generated],
        "query_records_checked": int(
            tr.groupby(["shot", "dataset"]).n_queries.first().sum()
        ),
        "paired_comparisons_checked": len(primary),
        "boundary_bins": len(bn),
        "scope": "Regeneration from existing result records; no model training.",
    }
    (out / "generation_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Done: {len(generated)} current figures at {dpi} DPI in {out}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=None,
        help="Results folder (auto-detected if omitted)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output folder (default: figures/ in the repository root)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PNG resolution (default: 300)"
    )
    parser.add_argument("--pdf", action="store_true", help="Also generate vector PDFs")
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    try:
        results = resolve_results(args.results)
        out = (
            (args.out or Path(__file__).resolve().parents[1] / "figures")
            .expanduser()
            .resolve()
        )
        generate(results, out, args.dpi, args.pdf)
    except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
        parser.exit(
            1,
            f"Figure generation stopped: {exc or 'Result-consistency check failed; use the matched result records.'}\n",
        )


if __name__ == "__main__":
    main()
