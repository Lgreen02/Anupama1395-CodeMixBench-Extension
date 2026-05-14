"""
Option B — Cross-Family Degradation Analysis
============================================
Computes degradation ratios per (model, language pair), correlates them
with phylogenetic / WALS linguistic distance, and produces:
  1. A heatmap of degradation ratios (models × language pairs)
  2. A scatter plot of degradation ratio vs. linguistic distance
     with Spearman correlation coefficient and p-value

No new model runs required — works entirely from existing *_metric.csv
files produced by test_model.py, plus the accuracy tables from the paper
(Table 2) which are embedded as constants for reference.

Requirements:
    pip install pandas matplotlib seaborn scipy numpy

Usage:
    # With your own replicated result CSVs:
    python option_b_degradation_analysis.py --result_dir result/

    # Or using paper's Table 2 numbers only (no CSV files needed):
    python option_b_degradation_analysis.py --use_paper_numbers
"""

import os
import re
import argparse
import logging
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paper Table 2 — CM-MMLU one-shot accuracy (embedded as ground truth)
# Format: {model_name: {lang_pair: accuracy}}
# ---------------------------------------------------------------------------
PAPER_TABLE2_CMMMLU = {
    "GPT-3.5-Instruct": {
        "en-only": 64.90, "zh-en": 60.99, "hi-en": 53.32, "bn-en": 46.32,
        "mr-en": 46.95, "ne-en": 46.70, "es-en": 65.01, "fr-en": 67.21,
        "ar-en": 53.94, "ta-en": 44.03, "nl-en": 66.08, "de-en": 67.63,
    },
    "GPT-3.5-Turbo": {
        "en-only": 66.30, "zh-en": 60.81, "hi-en": 55.37, "bn-en": 47.49,
        "mr-en": 49.67, "ne-en": 48.78, "es-en": 69.20, "fr-en": 68.83,
        "ar-en": 56.45, "ta-en": 45.75, "nl-en": 67.14, "de-en": 68.46,
    },
    "GPT-4-Turbo": {
        "en-only": 83.10, "zh-en": 79.08, "hi-en": 77.83, "bn-en": 72.26,
        "mr-en": 72.26, "ne-en": 72.78, "es-en": 81.24, "fr-en": 81.21,
        "ar-en": 77.06, "ta-en": 64.09, "nl-en": 82.64, "de-en": 80.71,
    },
    "GPT-4o": {
        "en-only": 85.60, "zh-en": 82.97, "hi-en": 82.13, "bn-en": 78.28,
        "mr-en": 77.98, "ne-en": 76.70, "es-en": 86.30, "fr-en": 85.28,
        "ar-en": 80.35, "ta-en": 70.77, "nl-en": 85.37, "de-en": 84.60,
    },
    "LLaMA2-7B": {
        "en-only": 38.00, "zh-en": 30.80, "hi-en": 29.00, "bn-en": 25.49,
        "mr-en": 29.05, "ne-en": 25.91, "es-en": 32.37, "fr-en": 34.78,
        "ar-en": 25.71, "ta-en": 26.65, "nl-en": 32.60, "de-en": 34.32,
    },
    "LLaMA3-70B": {
        "en-only": 77.20, "zh-en": 73.79, "hi-en": 74.61, "bn-en": 69.75,
        "mr-en": 67.29, "ne-en": 66.52, "es-en": 79.67, "fr-en": 78.50,
        "ar-en": 71.86, "ta-en": 62.18, "nl-en": 79.30, "de-en": 77.18,
    },
    "Mixtral-8x22B": {
        "en-only": 75.50, "zh-en": 66.64, "hi-en": 59.67, "bn-en": 55.03,
        "mr-en": 55.39, "ne-en": 55.13, "es-en": 74.17, "fr-en": 73.98,
        "ar-en": 59.83, "ta-en": 52.34, "nl-en": 71.98, "de-en": 72.54,
    },
}

CM_LANG_PAIRS = [
    "zh-en", "hi-en", "bn-en", "mr-en", "ne-en",
    "es-en", "fr-en", "ar-en", "ta-en", "nl-en", "de-en",
]

# ---------------------------------------------------------------------------
# Linguistic distance definitions
# ---------------------------------------------------------------------------

# Phylogenetic distance: categorical proxy
# 0 = same branch (Germanic-Germanic), 1 = same superfamily, 2 = cross-family
# English is Germanic (Indo-European). We measure distance from English.
PHYLO_DISTANCE = {
    # Indo-European, Germanic — closest
    "de-en": 0,   # German: Germanic
    "nl-en": 0,   # Dutch: Germanic
    "fr-en": 1,   # French: Romance (Indo-European but different branch)
    "es-en": 1,   # Spanish: Romance
    # Indo-European, Indo-Aryan — same family, distant branch
    "hi-en": 1,   # Hindi: Indo-Aryan (IE family)
    "bn-en": 1,   # Bengali: Indo-Aryan
    "mr-en": 1,   # Marathi: Indo-Aryan
    "ne-en": 1,   # Nepali: Indo-Aryan
    # Non-Indo-European — cross-family (distance 2)
    "zh-en": 2,   # Chinese: Sino-Tibetan
    "ar-en": 2,   # Arabic: Afro-Asiatic
    "ta-en": 2,   # Tamil: Dravidian
}

# WALS-derived approximate feature overlap scores (manually compiled from
# https://wals.info for the languages in the benchmark vs. English).
# Scale 0–1: higher = more similar to English structurally.
# These are approximate; replace with your own WALS computation if desired.
WALS_SIMILARITY = {
    "de-en": 0.72,
    "nl-en": 0.75,
    "fr-en": 0.62,
    "es-en": 0.61,
    "hi-en": 0.45,
    "bn-en": 0.43,
    "mr-en": 0.41,
    "ne-en": 0.44,
    "zh-en": 0.38,
    "ar-en": 0.33,
    "ta-en": 0.28,
}
# WALS distance = 1 - similarity
WALS_DISTANCE = {k: round(1 - v, 3) for k, v in WALS_SIMILARITY.items()}

# Language family labels for plot annotations
FAMILY_LABEL = {
    "de-en":  "Germanic",
    "nl-en":  "Germanic",
    "fr-en":  "Romance",
    "es-en":  "Romance",
    "hi-en":  "Indo-Aryan",
    "bn-en":  "Indo-Aryan",
    "mr-en":  "Indo-Aryan",
    "ne-en":  "Indo-Aryan",
    "zh-en":  "Sino-Tibetan",
    "ar-en":  "Afro-Asiatic",
    "ta-en":  "Dravidian",
}

FAMILY_COLOR = {
    "Germanic":     "#2196F3",
    "Romance":      "#4CAF50",
    "Indo-Aryan":   "#FF9800",
    "Sino-Tibetan": "#E91E63",
    "Afro-Asiatic": "#9C27B0",
    "Dravidian":    "#F44336",
}

# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------
def compute_degradation_ratio(en_only_acc: float, cm_acc: float) -> float:
    """
    degradation_ratio = (en_only - cm) / en_only
    0  = no degradation
    1  = complete collapse
    <0 = code-mixing actually helped (rare)
    """
    if en_only_acc == 0 or np.isnan(en_only_acc):
        return float("nan")
    return (en_only_acc - cm_acc) / en_only_acc


# ---------------------------------------------------------------------------
# Load results from CSV files (your own replication runs)
# ---------------------------------------------------------------------------
def load_metric_csvs(result_dir: str) -> dict:
    """
    Walk result_dir for *_metric.csv files.
    Expected filename pattern (from test_model.py):
        {model}_{lang_pair}_cm_mmlu_*_metric.csv
    Returns {model: {lang_pair: accuracy}} — same structure as PAPER_TABLE2_CMMMLU.
    """
    result_dir = Path(result_dir)
    pattern = re.compile(
        r"(?P<model>.+?)_(?P<lang>[a-z]+-[a-z]+)_cm_mmlu.*_metric\.csv",
        re.IGNORECASE,
    )
    acc_data = {}

    for fpath in result_dir.glob("**/*metric*.csv"):
        m = pattern.match(fpath.name)
        if not m:
            log.debug(f"Skipping unrecognized filename: {fpath.name}")
            continue
        model = m.group("model")
        lang = m.group("lang")
        try:
            df = pd.read_csv(fpath)
            # Look for an 'accuracy' column; fall back to first numeric column
            if "accuracy" in df.columns:
                acc = float(df["accuracy"].iloc[0])
            else:
                numeric_cols = df.select_dtypes(include="number").columns
                acc = float(df[numeric_cols[0]].iloc[0]) if len(numeric_cols) else float("nan")
            acc_data.setdefault(model, {})[lang] = acc
            log.info(f"Loaded {model} / {lang}: {acc:.2f}%")
        except Exception as e:
            log.warning(f"Could not parse {fpath}: {e}")

    return acc_data


# ---------------------------------------------------------------------------
# Build degradation dataframe
# ---------------------------------------------------------------------------
def build_degradation_df(acc_data: dict) -> pd.DataFrame:
    """
    acc_data: {model: {lang_pair: accuracy}}
    Returns a long-form DataFrame with columns:
      model, lang_pair, en_only_acc, cm_acc, degradation_ratio,
      phylo_distance, wals_distance, family
    """
    rows = []
    for model, lang_acc in acc_data.items():
        en_only = lang_acc.get("en-only", float("nan"))
        for lp in CM_LANG_PAIRS:
            cm_acc = lang_acc.get(lp, float("nan"))
            if np.isnan(cm_acc):
                continue
            rows.append({
                "model":             model,
                "lang_pair":         lp,
                "en_only_acc":       en_only,
                "cm_acc":            cm_acc,
                "degradation_ratio": compute_degradation_ratio(en_only, cm_acc),
                "phylo_distance":    PHYLO_DISTANCE.get(lp, float("nan")),
                "wals_distance":     WALS_DISTANCE.get(lp, float("nan")),
                "family":            FAMILY_LABEL.get(lp, "Unknown"),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------
def run_correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman correlation between linguistic distance and degradation ratio
    per model, then overall.
    Returns a summary DataFrame.
    """
    results = []
    for dist_col, dist_label in [("phylo_distance", "Phylogenetic"),
                                  ("wals_distance", "WALS")]:
        # Per-model correlations
        for model in df["model"].unique():
            sub = df[df["model"] == model].dropna(subset=[dist_col, "degradation_ratio"])
            if len(sub) < 4:
                continue
            rho, pval = spearmanr(sub[dist_col], sub["degradation_ratio"])
            results.append({
                "scope": model, "distance_type": dist_label,
                "spearman_rho": round(rho, 3), "p_value": round(pval, 4),
                "n": len(sub),
            })
        # Pooled across all models
        pooled = df.dropna(subset=[dist_col, "degradation_ratio"])
        rho, pval = spearmanr(pooled[dist_col], pooled["degradation_ratio"])
        results.append({
            "scope": "ALL MODELS (pooled)", "distance_type": dist_label,
            "spearman_rho": round(rho, 3), "p_value": round(pval, 4),
            "n": len(pooled),
        })

    corr_df = pd.DataFrame(results)
    print("\n=== Spearman Correlation: Linguistic Distance vs. Degradation Ratio ===")
    print(corr_df.to_string(index=False))
    return corr_df


# ---------------------------------------------------------------------------
# Figure 1: Degradation heatmap (models × language pairs)
# ---------------------------------------------------------------------------
def plot_degradation_heatmap(df: pd.DataFrame, output_path: str):
    pivot = df.pivot_table(
        index="model", columns="lang_pair", values="degradation_ratio"
    )
    # Reorder columns by average degradation (ascending = least degraded on left)
    col_order = pivot.mean(axis=0).sort_values().index.tolist()
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(max(12, len(col_order) * 1.2),
                                    max(4, len(pivot) * 0.7)))
    sns.heatmap(
        pivot,
        annot=True, fmt=".2f",
        cmap="RdYlGn_r",       # red = high degradation, green = low
        vmin=0, vmax=0.30,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Degradation Ratio\n(en_only − cm) / en_only"},
    )
    ax.set_title("Degradation Ratio: CM-MMLU One-shot vs. English-only\n"
                 "(Higher = more degraded by code-mixing)", fontsize=12, pad=12)
    ax.set_xlabel("Language Pair", fontsize=10)
    ax.set_ylabel("Model", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    # Add family annotation below x-axis
    for i, lp in enumerate(col_order):
        fam = FAMILY_LABEL.get(lp, "")
        ax.text(i + 0.5, len(pivot) + 0.7, fam, ha="center", va="bottom",
                fontsize=6, rotation=30, color="gray")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Heatmap saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: Scatter plot — degradation ratio vs. linguistic distance
# ---------------------------------------------------------------------------
def plot_degradation_scatter(df: pd.DataFrame, dist_col: str,
                             dist_label: str, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, scope, title in [
        (axes[0], "pooled", "All Models Pooled"),
        (axes[1], "per_lp", "Per Language Pair (avg across models)"),
    ]:
        if scope == "pooled":
            plot_df = df.dropna(subset=[dist_col, "degradation_ratio"]).copy()
        else:
            # Average degradation per language pair across models
            plot_df = (
                df.groupby("lang_pair")
                  .agg(degradation_ratio=("degradation_ratio", "mean"),
                       **{dist_col: (dist_col, "first")},
                       family=("family", "first"))
                  .reset_index()
                  .dropna(subset=[dist_col, "degradation_ratio"])
            )

        # Scatter points colored by family
        for fam, fam_df in plot_df.groupby("family"):
            ax.scatter(
                fam_df[dist_col], fam_df["degradation_ratio"],
                label=fam, color=FAMILY_COLOR.get(fam, "gray"),
                s=60, alpha=0.75, edgecolors="white", linewidths=0.5,
            )

        # Regression line
        x = plot_df[dist_col].values
        y = plot_df["degradation_ratio"].values
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), "k--", linewidth=1.2, alpha=0.6)

            rho, pval = spearmanr(x, y)
            ax.text(0.05, 0.92,
                    f"ρ = {rho:.3f}, p = {pval:.4f}",
                    transform=ax.transAxes, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Annotate language pair labels (per-lp plot only)
        if scope == "per_lp":
            for _, row in plot_df.iterrows():
                ax.annotate(
                    row["lang_pair"],
                    (row[dist_col], row["degradation_ratio"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=7,
                )

        ax.set_xlabel(f"{dist_label} Distance from English", fontsize=10)
        ax.set_ylabel("Degradation Ratio", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(linestyle=":", alpha=0.4)

    fig.suptitle(
        f"Degradation Ratio vs. {dist_label} Distance (CM-MMLU)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Scatter plot saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: Same-family vs. cross-family bar chart
# ---------------------------------------------------------------------------
def plot_family_comparison(df: pd.DataFrame, output_path: str):
    """Bar chart: average degradation for same-family vs. cross-family pairs."""
    df = df.copy()
    df["cross_family"] = df["phylo_distance"].apply(
        lambda d: "Cross-family\n(zh-en, ar-en, ta-en)" if d == 2
        else "Same-superfamily\n(hi-en, bn-en, mr-en, ne-en)" if d == 1
        else "Same-branch\n(de-en, nl-en, fr-en, es-en)"
    )

    order = [
        "Same-branch\n(de-en, nl-en, fr-en, es-en)",
        "Same-superfamily\n(hi-en, bn-en, mr-en, ne-en)",
        "Cross-family\n(zh-en, ar-en, ta-en)",
    ]

    agg = (df.groupby(["model", "cross_family"])["degradation_ratio"]
             .mean()
             .reset_index())
    agg["cross_family"] = pd.Categorical(agg["cross_family"], categories=order, ordered=True)
    agg = agg.sort_values("cross_family")

    fig, ax = plt.subplots(figsize=(10, 5))
    models = df["model"].unique()
    x = np.arange(len(order))
    width = 0.8 / max(len(models), 1)
    colors = plt.cm.Set2.colors

    for i, model in enumerate(models):
        sub = agg[agg["model"] == model].set_index("cross_family")
        vals = [sub.loc[cat, "degradation_ratio"] if cat in sub.index else np.nan
                for cat in order]
        ax.bar(x + i * width - (len(models) - 1) * width / 2,
               vals, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel("Mean Degradation Ratio", fontsize=10)
    ax.set_title("Code-Mixing Degradation by Linguistic Proximity to English",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_ylim(0, max(0.3, agg["degradation_ratio"].max() * 1.2))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    log.info(f"Family comparison chart saved to {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Option B: Cross-family degradation analysis")
    p.add_argument("--result_dir", default="result/",
                   help="Directory containing *_metric.csv files from test_model.py")
    p.add_argument("--output_dir", default="result/option_b",
                   help="Where to save figures and tables")
    p.add_argument("--use_paper_numbers", action="store_true",
                   help="Use Table 2 numbers from the paper instead of loading CSVs")
    p.add_argument("--dist_measure", choices=["phylo", "wals", "both"], default="both",
                   help="Linguistic distance measure to use for scatter plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load accuracy data
    if args.use_paper_numbers:
        log.info("Using paper Table 2 numbers (--use_paper_numbers)")
        acc_data = PAPER_TABLE2_CMMMLU
    else:
        log.info(f"Loading metric CSVs from: {args.result_dir}")
        acc_data = load_metric_csvs(args.result_dir)
        if not acc_data:
            log.warning("No CSV files found. Falling back to paper numbers.")
            acc_data = PAPER_TABLE2_CMMMLU

    # Build degradation dataframe
    df = build_degradation_df(acc_data)
    if df.empty:
        log.error("Degradation dataframe is empty. Check your result_dir or CSV format.")
        return

    log.info(f"\nDegradation DataFrame shape: {df.shape}")
    print("\n=== Sample rows ===")
    print(df.head(10).to_string(index=False))

    # Save raw degradation table
    raw_path = os.path.join(args.output_dir, "degradation_table.csv")
    df.to_csv(raw_path, index=False)
    log.info(f"Raw degradation table saved to {raw_path}")

    # Correlation analysis
    corr_df = run_correlation_analysis(df)
    corr_path = os.path.join(args.output_dir, "correlation_results.csv")
    corr_df.to_csv(corr_path, index=False)

    # Figure 1 — Heatmap
    plot_degradation_heatmap(
        df,
        output_path=os.path.join(args.output_dir, "fig_b1_degradation_heatmap.png"),
    )

    # Figure 2 — Scatter plots
    dist_configs = []
    if args.dist_measure in ("phylo", "both"):
        dist_configs.append(("phylo_distance", "Phylogenetic (Categorical)"))
    if args.dist_measure in ("wals", "both"):
        dist_configs.append(("wals_distance", "WALS Feature"))

    for dist_col, dist_label in dist_configs:
        safe_label = dist_col.replace("_", "")
        plot_degradation_scatter(
            df, dist_col=dist_col, dist_label=dist_label,
            output_path=os.path.join(args.output_dir,
                                     f"fig_b2_scatter_{safe_label}.png"),
        )

    # Figure 3 — Family comparison bar chart
    plot_family_comparison(
        df,
        output_path=os.path.join(args.output_dir, "fig_b3_family_comparison.png"),
    )

    # Print paper-ready summary
    summary = (
        df.groupby("lang_pair")
          .agg(
              mean_degradation=("degradation_ratio", "mean"),
              phylo_distance=("phylo_distance", "first"),
              wals_distance=("wals_distance", "first"),
              family=("family", "first"),
          )
          .sort_values("mean_degradation", ascending=False)
          .reset_index()
    )
    print("\n=== Language-pair Summary (sorted by degradation) ===")
    print(summary.to_string(index=False))
    summary.to_csv(os.path.join(args.output_dir, "langpair_summary.csv"), index=False)

    log.info(f"\nAll outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()