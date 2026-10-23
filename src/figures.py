"""Figures for the results write-up.

Every figure reads the artefacts the modelling scripts leave behind, so running
this file never re-estimates anything and the pictures cannot drift away from
the numbers in the tables beside them.

Colour follows one rule throughout: hues are assigned by identity, in a fixed
order, and never cycled. Magnitude is carried by position and length rather
than by colour. Each figure has a Markdown table next to it in
``results/tables``, which is what makes the lighter hues legible for readers
who cannot separate them.

Usage
-----
    python src/figures.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import validation as val  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

# Chart surface and ink, kept apart from the series hues: text never wears a
# series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

# Categorical slots, in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]

SAMPLES = ["estimation", "out of sample", "out of time"]
DPI = 160


def style() -> None:
    """Recessive axes, thin marks, generous white space."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.9,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.size": 10,
            "figure.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    path = FIGURES / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"  {name}")
    return path


def load_scored() -> pd.DataFrame:
    path = PROCESSED / "loans_with_ecl.csv"
    if not path.exists():
        path = PROCESSED / "loans_with_pd.csv"
    if not path.exists():
        raise FileNotFoundError("run src/pd_model.py and src/lgd_ead_model.py first")
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# 1. Discrimination
# ---------------------------------------------------------------------------
def figure_roc(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.56, "no discrimination", color=INK_SOFT, fontsize=8.5, rotation=36,
            rotation_mode="anchor", ha="left", va="bottom")

    for index, sample in enumerate(SAMPLES):
        block = data[data["sample"] == sample]
        fpr, tpr, _ = roc_curve(block["default"], block["pd_predicted"])
        auc = val.discrimination(block["default"], block["pd_predicted"])["auc"]
        ax.plot(fpr, tpr, color=SERIES[index], linewidth=2.0, zorder=3,
                label=f"{sample}  (AUC {auc:.3f})")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("Discrimination of the probability-of-default model")
    ax.grid(True, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    save(fig, "roc_curves.png")


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------
def figure_calibration(data: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    limit = 0.0

    for index, sample in enumerate(SAMPLES):
        block = data[data["sample"] == sample]
        table = val.calibration_table(block["default"], block["pd_predicted"])
        limit = max(limit, table[["predicted", "observed"]].to_numpy().max())
        ax.plot(table["predicted"], table["observed"], color=SERIES[index],
                linewidth=1.8, marker="o", markersize=7, markeredgecolor=SURFACE,
                markeredgewidth=1.4, zorder=3, label=sample)

    span = [0, limit * 1.08]
    ax.plot(span, span, color=GRID, linewidth=1.6, linestyle=(0, (4, 3)), zorder=1)
    # Below the diagonal, where no series runs.
    ax.text(span[1] * 0.74, span[1] * 0.60, "perfect calibration", color=INK_SOFT,
            fontsize=8.5, rotation=45, rotation_mode="anchor", ha="center", va="top")
    ax.text(span[1] * 0.30, span[1] * 0.88,
            "above the line: the model quotes\nless default than the book delivers",
            color=INK_SOFT, fontsize=8.5, ha="left", va="top")

    ax.set_xlim(span)
    ax.set_ylim(span)
    ax.set_xlabel("mean quoted probability of default, by decile")
    ax.set_ylabel("default rate actually observed")
    ax.set_title("Calibration: quoted against observed, by decile of risk")
    ax.grid(True, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    save(fig, "calibration.png")


# ---------------------------------------------------------------------------
# 3. Information value
# ---------------------------------------------------------------------------
def figure_information_values() -> None:
    path = TABLES / "information_values.csv"
    if not path.exists():
        return
    table = pd.read_csv(path).sort_values("iv")

    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    positions = np.arange(len(table))
    retained = table["iv"] >= 0.01
    colours = [BLUE if keep else GRID for keep in retained]

    ax.barh(positions, table["iv"], height=0.68, color=colours, zorder=3)
    for y, value in zip(positions, table["iv"]):
        ax.text(value + 0.0022, y, f"{value:.3f}", va="center", ha="left",
                fontsize=8.5, color=INK_SOFT)

    ax.axvline(0.01, color=ORANGE, linewidth=1.6, linestyle=(0, (4, 3)), zorder=4)
    ax.text(0.0115, len(table) - 0.4, "screening floor, 0.01", color=ORANGE, fontsize=8.5)

    ax.set_yticks(positions)
    ax.set_yticklabels(table["predictor"], fontsize=9)
    ax.set_xlim(0, float(table["iv"].max()) * 1.22)
    ax.set_xlabel("information value")
    ax.set_title("Univariate screening: information value of each predictor")
    ax.grid(True, axis="x", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "information_values.png")


# ---------------------------------------------------------------------------
# 4. Score separation
# ---------------------------------------------------------------------------
def figure_score_distribution(data: pd.DataFrame) -> None:
    block = data[data["sample"] == "out of sample"]
    repaid = block.loc[block["default"] == 0, "score"]
    defaulted = block.loc[block["default"] == 1, "score"]

    low = float(np.nanpercentile(block["score"], 0.2))
    high = float(np.nanpercentile(block["score"], 99.8))
    edges = np.linspace(low, high, 46)

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.hist(repaid, bins=edges, density=True, color=BLUE, alpha=0.62,
            label="repaid in full", zorder=3)
    ax.hist(defaulted, bins=edges, density=True, color=ORANGE, alpha=0.62,
            label="charged off", zorder=3)

    for series, colour, offset in ((repaid, BLUE, 1.06), (defaulted, ORANGE, 0.94)):
        mean = float(series.mean())
        ax.axvline(mean, color=colour, linewidth=1.8, linestyle=(0, (4, 3)), zorder=4)

    ks = val.ks_statistic(block["default"], block["pd_predicted"])
    ax.set_title(f"Score separation on the out-of-sample part "
                 f"(Kolmogorov-Smirnov {ks:.3f})")
    ax.set_xlabel("scorecard points")
    ax.set_ylabel("density")
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    save(fig, "score_distribution.png")


# ---------------------------------------------------------------------------
# 5. Expected against realised loss
# ---------------------------------------------------------------------------
def figure_loss_by_band() -> None:
    path = TABLES / "ecl_by_score_band.csv"
    if not path.exists():
        return
    table = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    positions = np.arange(len(table))
    width = 0.38
    gap = 0.012  # surface gap between adjacent bars

    ax.bar(positions - width / 2 - gap, table["expected_loss_pct"], width,
           color=BLUE, label="expected credit loss", zorder=3)
    ax.bar(positions + width / 2 + gap, table["realised_loss_pct"], width,
           color=ORANGE, label="loss actually borne", zorder=3)

    for x, (expected, realised) in enumerate(
        zip(table["expected_loss_pct"], table["realised_loss_pct"])
    ):
        ax.text(x - width / 2 - gap, expected + 0.45, f"{expected:.1f}",
                ha="center", fontsize=8.5, color=INK_SOFT)
        ax.text(x + width / 2 + gap, realised + 0.45, f"{realised:.1f}",
                ha="center", fontsize=8.5, color=INK_SOFT)

    ax.set_xticks(positions)
    ax.set_xticklabels(table["score_band"])
    ax.set_xlabel("scorecard band")
    ax.set_ylabel("per cent of exposure")
    ax.set_ylim(0, max(table["expected_loss_pct"].max(), table["realised_loss_pct"].max()) * 1.18)
    ax.set_title("Expected credit loss against the loss actually borne, by scorecard band")
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    save(fig, "expected_vs_realised_loss.png")


# ---------------------------------------------------------------------------
# 6. The drift the out-of-time sample is there to catch
# ---------------------------------------------------------------------------
def figure_default_rate_by_vintage(data: pd.DataFrame) -> None:
    by_year = data.groupby("issue_year").agg(
        observed=("default", "mean"), loans=("default", "size")
    ).reset_index()
    by_year = by_year[by_year["loans"] >= 500]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.plot(by_year["issue_year"], 100 * by_year["observed"], color=BLUE,
            linewidth=2.0, marker="o", markersize=8, markeredgecolor=SURFACE,
            markeredgewidth=1.6, zorder=3)

    for year, rate in zip(by_year["issue_year"], 100 * by_year["observed"]):
        ax.text(year, rate + 0.45, f"{rate:.1f}", ha="center", fontsize=8.5, color=INK_SOFT)

    last = int(by_year["issue_year"].max())
    ax.axvspan(2014.5, last + 0.45, color=ORANGE, alpha=0.09, zorder=1)
    # Anchored inside the band and below its top edge, so it cannot run off
    # the right-hand side however the limits end up.
    low, high = ax.get_ylim()
    ax.text(last + 0.35, high - 0.05 * (high - low), "out-of-time\nsample",
            color=ORANGE, fontsize=8.5, ha="right", va="top", linespacing=1.35)

    ax.set_xticks(by_year["issue_year"].astype(int))
    ax.set_xlabel("year of issuance")
    ax.set_ylabel("observed default rate, per cent")
    ax.set_title("The book the model was fitted on is not the book it then meets")
    ax.grid(True, axis="y", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, "default_rate_by_vintage.png")


def main() -> int:
    FIGURES.mkdir(parents=True, exist_ok=True)
    style()
    data = load_scored()

    print("Figures written to results/figures:")
    figure_roc(data)
    figure_calibration(data)
    figure_information_values()
    figure_score_distribution(data)
    figure_loss_by_band()
    figure_default_rate_by_vintage(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
