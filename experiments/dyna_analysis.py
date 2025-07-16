from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from rliable.metrics import probability_of_improvement
from scipy.stats import (
    f_oneway,
    kruskal,
    levene,
    mannwhitneyu,
    sem,
    shapiro,
    trim_mean,
    ttest_ind,
    ttest_rel,
    wilcoxon,
)
from sklearn.utils import resample
from statsmodels.stats.multitest import multipletests

from experiments.uid_data import ablation_uids, main_uids, paradigm_uids
from marloes.results.calculator import Calculator

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

colors_ablation = [
    ("rgb(0, 158, 115)", "solid"),  # green for with GRU
    ("rgb(213, 94, 0)", "solid"),  # red for no GRU
]

colors_paradigm = [
    ("rgb(0, 158, 115)", "solid"),
    ("rgb(0, 158, 115)", "dash"),
    ("rgb(0, 158, 115)", "dot"),
    ("rgb(0, 114, 178)", "solid"),
    ("rgb(0, 114, 178)", "dash"),
    ("rgb(0, 114, 178)", "dot"),
]

colors_main = [
    # yellow for PrioFlow
    ("rgb(230, 159, 0)", "solid"),
    ("rgb(86, 180, 233)", "solid"),  # blue for SAC
    ("rgb(0, 158, 115)", "solid"),  # green for DynaSAC (rho=0.8)
    ("rgb(204, 121, 167)", "solid"),  # orange for DynaSAC (rho=0.5)
]
EXPERIMENTS = {
    "ablation": ablation_uids,
    "paradigm": paradigm_uids,
    "main_zero_noise": main_uids["zero_noise"],
    "main_default": main_uids["default"],
}


def do_bootstrap_ci(data):
    """
    Returns the (1–alpha) CI of func(data) via bootstrap.
    """
    stats = []
    n_bootstraps = 5000
    alpha = 0.05
    for _ in range(n_bootstraps):
        sample = resample(data, replace=True)
        stats.append(np.mean(sample))
    lower = np.percentile(stats, 100 * (alpha / 2))
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return lower, upper


##  Extractor functions
def _series_for_uid(uid: int, metric: str, scenario: str, rolling: bool) -> pd.Series:
    """Return metric as a Series with DateTimeIndex for one UID."""
    calc = Calculator(
        uid, dir="results" if scenario == "training" else f"evaluate/{scenario}"
    )
    s = calc.get_metrics([metric])[metric]
    start = calc.get_metrics(["start_time"])["start_time"]
    if isinstance(s, np.ndarray):
        s = pd.Series(
            s,
            index=range(0, len(s)),
        )
    if rolling:
        s = s.rolling(window=60, min_periods=1).median()
    return s


def build_group_dataframe(
    uid_list: List[int], metric: str, scenario: str, rolling: bool
) -> pd.DataFrame:
    """DataFrame rows = time index, columns = individual runs."""
    data = {uid: _series_for_uid(uid, metric, scenario, rolling) for uid in uid_list}
    union_idx = pd.Index([])
    for s in data.values():
        union_idx = union_idx.union(s.index)
    return pd.DataFrame({k: s.reindex(union_idx) for k, s in data.items()})


## Plotting/summary
def plot_groups_ci(
    groups: Dict[str, List[int]],
    metric: str,
    scenario: str,
    rolling: bool,
    save_png: bool,
    experiment: str,
):
    """Plot every group’s mean ±95 % CI in one figure."""
    if experiment == "ablation":
        colors = colors_ablation
    elif experiment == "paradigm":
        colors = colors_paradigm
    elif experiment == "main_zero_noise" or experiment == "main_default":
        colors = colors_main
    else:
        colors = [
            "rgb(0, 158, 115)",
            "rgb(213, 94, 0)",
            "rgb(230, 159, 0)",
            "rgb(86, 180, 233)",
            "rgb(240, 228, 66)",
            "rgb(0, 114, 178)",
            "rgb(204, 121, 167)",
        ]
        colors = [(c, "solid") for c in colors]  # Default line type

    fig = go.Figure()
    mean_lines = []
    for idx, (label, uids) in enumerate(groups.items()):
        df = build_group_dataframe(uids, metric, scenario, rolling)
        mean = df.mean(axis=1)
        ci = 1.96 * df.apply(sem, axis=1, nan_policy="omit")
        color, line_type = colors[idx % len(colors)]

        # CI band
        fig.add_trace(
            go.Scatter(
                x=mean.index.append(mean.index[::-1]),
                y=(mean - ci)._append((mean + ci)[::-1]),
                fill="toself",
                fillcolor=color.replace("rgb", "rgba").replace(")", ",0.18)"),
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # CI outline (upper and lower boundaries)
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean - ci,
                mode="lines",
                line=dict(
                    color=color.replace("rgb", "rgba").replace(
                        ")", ",0.5)"
                    ),  # half-opaque border
                    width=2,
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean + ci,
                mode="lines",
                line=dict(
                    color=color.replace("rgb", "rgba").replace(")", ",0.5)"), width=2
                ),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        # Mean line
        mean_lines.append(
            go.Scatter(
                x=mean.index,
                y=mean,
                mode="lines",
                line=dict(width=3, color=color, dash=line_type),
                name=f"{label}",
            )
        )

    # Add mean lines to figure
    for line in mean_lines:
        fig.add_trace(line)

    if experiment == "paradigm":
        title = "Cumulative CO₂ emissions – DynaSAC and DynaMASAC (3, 6, 12 assets)"
    else:
        title = f"Cumulative CO₂ emissions – {', '.join(groups.keys())}"
    if experiment == "main_default":
        title += " - Noisy data"

    fig.update_layout(
        title=title,
        xaxis_title="Time-steps (minutes)",
        yaxis_title="Cumulative CO₂ emissions (gCO₂eq)",
        font=dict(family="Times New Roman, Times, serif", size=35),
        legend=dict(font=dict(size=35)),
        title_font=dict(size=45),
    )
    fig.show()

    if save_png:
        outdir = f"results/img/{metric}"
        os.makedirs(outdir, exist_ok=True)
        fig.write_image("my_figure.pdf", format="pdf", width=800, height=600, scale=4)


def summary(
    groups: Dict[str, List[int]],
    metric: str,
    scenario: str,
) -> pd.DataFrame:
    """Return table of mean ±95 % CI for final value."""
    rows_performance = []
    rows_uncertainty = []
    for label, uids in groups.items():
        vals = []
        for uid in uids:
            calc = Calculator(
                uid, dir="results" if scenario == "training" else f"evaluate/{scenario}"
            )
            vals.append(calc.get_metrics([metric])[metric][-1])

        # Performance metrics
        vals = np.asarray(vals)
        mean = vals.mean()
        ci = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
        median = np.median(vals)
        iqm = trim_mean(vals, proportiontocut=0.25)  # Interquartile mean

        # Uncertainty metrics
        cv = vals.std(ddof=1) / mean if mean != 0 else np.nan
        bootstrap_ci = do_bootstrap_ci(vals)

        rows_performance.append((label, mean, ci, median, iqm))
        rows_uncertainty.append((label, cv, bootstrap_ci[0], bootstrap_ci[1]))

    df_performance = pd.DataFrame(
        rows_performance, columns=["group", "mean", "ci95", "median", "iqm"]
    )
    df_uncertainty = pd.DataFrame(
        rows_uncertainty,
        columns=["group", "cv", "bootstrap_ci_lower", "bootstrap_ci_upper"],
    )

    print("\n=== Performance Summary ===")
    print(df_performance.to_string(index=False))

    print("\n=== Uncertainty Summary ===")
    print(df_uncertainty.to_string(index=False))


## Statistical tests


def extract_final_metric(uids, scenario):
    vals = []
    for uid in uids:
        calc = Calculator(
            uid, dir="results" if scenario == "training" else f"evaluate/{scenario}"
        )
        vals.append(
            calc.get_metrics(["cumulative_co2_emissions"])["cumulative_co2_emissions"][
                -1
            ]
        )
    return np.array(vals)


def cohens_d(x, y):
    """Cohen's d for paired samples."""
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


def t_test_summary(
    group1, group2, scenario, label1="A", label2="B", vals1=None, vals2=None
):
    """
    Performs Shapiro–Wilk test, paired t-test or Wilcoxon, and Cohen's d.
    """
    if vals1 is None or vals2 is None:
        vals1 = extract_final_metric(group1, scenario)
        vals2 = extract_final_metric(group2, scenario)
    diff = vals1 - vals2

    # Normality test
    shapiro_p = shapiro(diff).pvalue

    if shapiro_p >= 0.05:
        stat, p = ttest_rel(vals1, vals2)
        test = "paired t-test"
    else:
        stat, p = wilcoxon(vals1, vals2)
        test = "Wilcoxon signed-rank"

    d = cohens_d(vals1, vals2)

    scores_x = vals1.reshape(-1, 1)
    scores_y = vals2.reshape(-1, 1)

    poi = probability_of_improvement(scores_x, scores_y)

    logging.info(
        f"{test}: statistic={stat:.3f}, p={p:.3g} | Shapiro p={shapiro_p:.3g} | Cohen's d={d:.3f}"
    )
    print(f"\n--- Statistical test: {label1} vs. {label2} ---")
    print(f"Test: {test}")
    print(f"Test statistic: {stat:.3f}")
    print(f"p-value: {p:.3g}")
    print(f"Shapiro-Wilk p-value for normality: {shapiro_p:.3g}")
    print(f"Cohen's d: {d:.3f}")
    print(f"Probability of Improvement: {poi:.3f}")

    return stat, p, d


def run_multiple_tests_and_apply_holm(
    test_pairs, groups, scenario, all_groups=None, all_labels=None
):
    # First do ANOVA to check if there are significant differences
    # between any of the groups
    if all_groups is None:
        all_groups = [extract_final_metric(groups[label], scenario) for label in groups]
        all_labels = list(groups.keys())

    # First check for normality for ANOVA
    all_vals = np.concatenate(all_groups)
    shapiro_p = shapiro(all_vals).pvalue

    print(f"\n== Shapiro-Wilk test for normality: p={shapiro_p:.3g} ==")

    # Do ANOVA/Kruskal-Wallis test
    if shapiro_p >= 0.05:
        stat, p = f_oneway(*all_groups)
        overall_test = "ANOVA"
    else:
        stat, p = kruskal(*all_groups)
        overall_test = "Kruskal–Wallis"

    print(f"\n== Overall test: {overall_test} ==")
    print(f"{overall_test} statistic={stat:.3f}, p-value={p:.4g}")

    # If overall not significant, return early
    if p >= 0.05:
        print(
            "No significant overall difference between groups (p >= 0.05). Skipping pairwise tests."
        )
        return

    raw_pvals = []
    results = []
    for label1, label2 in test_pairs:
        stat, p, cohen_d = t_test_summary(
            groups[label1],
            groups[label2],
            scenario,
            label1=label1,
            label2=label2,
        )
        raw_pvals.append(p)
        results.append((label1, label2, stat, p, cohen_d))
    # Holm–Bonferroni correction
    print("\n=== Holm–Bonferroni Correction ===")
    reject, pvals_holm, _, _ = multipletests(raw_pvals, method="holm")
    for i, (label1, label2, stat, raw_p, cohen_d) in enumerate(results):
        adj_p = pvals_holm[i]
        print(
            f"{label1} vs {label2}: t={stat:.2f}, raw p={raw_p:.3g}, Holm p={adj_p:.3g}, Cohen's d={cohen_d:.2f}, significant={reject[i]}"
        )


def efficiency_summary(
    groups: dict,
    pairs: list = None,
    scenario: str = "default",
) -> pd.DataFrame:
    """Return table of wall time and AULC for each group."""
    steps_per_day = 1440
    rows = []
    wall_data = {}
    aulc_data = {}

    for label, uids in groups.items():
        wall_times = []
        aulcs = []
        for uid in uids:
            calc = Calculator(uid, dir="results")
            # Wall time in hours
            wall_time = calc.get_metrics(["elapsed_time"])["elapsed_time"][-1] / 3600
            wall_times.append(wall_time)
            # Emissions for AULC
            emissions = np.array(calc.get_metrics(["co2_emissions"])["co2_emissions"])
            n_days = len(emissions) // steps_per_day
            daily_emissions = emissions[: n_days * steps_per_day].reshape(
                n_days, steps_per_day
            )
            mean_per_day = daily_emissions.mean(axis=1)
            aulc = np.trapz(mean_per_day)
            aulcs.append(aulc)

        # Summarize
        wall_times = np.array(wall_times)
        aulcs = np.array(aulcs)
        wall_data[label] = wall_times
        aulc_data[label] = aulcs

        def mean_ci(arr):
            m = arr.mean()
            ci = 1.96 * arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else np.nan
            return m, ci

        wall_mean, wall_ci = mean_ci(wall_times)
        aulc_mean, aulc_ci = mean_ci(aulcs)
        rows.append((label, wall_mean, wall_ci, aulc_mean, aulc_ci))

    df = pd.DataFrame(
        rows,
        columns=[
            "group",
            "wall_time_mean_hr",
            "wall_time_95%CI",
            "AULC_mean",
            "AULC_95%CI",
        ],
    )
    print("\n=== Efficiency Summary ===")
    print(df.to_string(index=False))

    # Do statistical tests if pairs are provided
    if pairs:
        print("\n=== Wall time pairwise tests ===")
        for l1, l2 in pairs:
            t_test_summary(
                None,
                None,
                scenario,
                label1=l1,
                label2=l2,
                vals1=wall_data[l1],
                vals2=wall_data[l2],
            )
        print("\n=== AULC pairwise tests ===")
        for l1, l2 in pairs:
            t_test_summary(
                None,
                None,
                scenario,
                label1=l1,
                label2=l2,
                vals1=aulc_data[l1],
                vals2=aulc_data[l2],
            )
    elif len(groups) == 2:
        # else two groups
        (l1, u1), (l2, u2) = groups.items()
        print(f"\n=== Pairwise tests for {l1} vs {l2} ===")
        print("Wall time:")
        t_test_summary(
            None,
            None,
            scenario,
            label1=l1,
            label2=l2,
            vals1=wall_data[l1],
            vals2=wall_data[l2],
        )
        print("AULC:")
        t_test_summary(
            None,
            None,
            scenario,
            label1=l1,
            label2=l2,
            vals1=aulc_data[l1],
            vals2=aulc_data[l2],
        )

    return df


def main():
    parser = argparse.ArgumentParser(description="Plot & stats for MARLOES experiments")
    parser.add_argument(
        "-e",
        "--experiment",
        required=True,
        choices=EXPERIMENTS.keys(),
        help="ablation | paradigm | main_zero_noise | main_default",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        default="default",
        help="Scenario to analyze (default: 'default')",
    )
    parser.add_argument(
        "--save-png",
        action="store_true",
        help="Save the plot as a PNG file (default: False)",
    )

    args = parser.parse_args()

    metric = "cumulative_co2_emissions"

    groups = EXPERIMENTS[args.experiment]
    empty = [lab for lab, u in groups.items() if not u]
    if empty:
        logging.error(f"UID list empty for: {', '.join(empty)}")
        return

    # Set log level to warning to avoid cluttering output
    logging.getLogger().setLevel(logging.WARNING)
    # Plot
    plot_groups_ci(
        groups,
        metric=metric,
        scenario=args.scenario,
        rolling=True,
        save_png=args.save_png,
        experiment=args.experiment,
    )

    # Summary table
    summary(groups, metric, args.scenario)

    # Stats
    print("\n=== Statistical tests ===")
    efficiency_pairs = None
    if args.experiment == "ablation":
        (label1, uids1), (label2, uids2) = list(groups.items())
        t_test_summary(uids1, uids2, args.scenario, label1, label2)
    elif args.experiment == "paradigm":
        pairs = [
            (f"DynaMASAC ({n} assets)", f"DynaSAC ({n} assets)")
            for n in [3, 6, 12]
            if f"DynaMASAC ({n} assets)" in groups and f"DynaSAC ({n} assets)" in groups
        ]
        efficiency_pairs = pairs
        run_multiple_tests_and_apply_holm(pairs, groups, args.scenario)
    elif args.experiment in ["main_zero_noise", "main_default"]:
        pairs = [
            ("SAC", "PrioFlow"),
            ("DynaSAC (rho=0.8)", "SAC"),
            ("DynaSAC (rho=0.5)", "SAC"),
            ("DynaSAC (rho=0.8)", "PrioFlow"),
            ("DynaSAC (rho=0.5)", "PrioFlow"),
            ("DynaSAC (rho=0.8)", "DynaSAC (rho=0.5)"),
        ]
        efficiency_pairs = [
            ("SAC", "DynaSAC (rho=0.8)"),
        ]
        run_multiple_tests_and_apply_holm(pairs, groups, args.scenario)

    # Efficiency summary
    efficiency_summary(groups, pairs=efficiency_pairs, scenario=args.scenario)


if __name__ == "__main__":
    main()
