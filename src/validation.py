"""Validation measures for a probability-of-default model.

The measures fall into three groups, and a model has to pass all three. They
are independent of one another: a model can rank borrowers perfectly and still
quote the wrong level, and one that is accurate today can drift away from the
population it was fitted on.

Discrimination
    Does the model put the defaults above the survivors? Area under the ROC
    curve, the Gini coefficient and the Kolmogorov-Smirnov distance.

Calibration
    Does the quoted probability match the default rate actually observed?
    The Hosmer-Lemeshow statistic, the Brier score and a decile table.

Stability
    Does the population still resemble the one the model was fitted on?
    The population stability index.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# Conventional reading of the population stability index.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25


def gini(auc: float) -> float:
    """Gini coefficient implied by an area under the ROC curve."""
    return 2.0 * auc - 1.0


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Largest gap between the cumulative score distributions of the two groups."""
    false_positive, true_positive, _ = roc_curve(y_true, y_score)
    return float(np.max(true_positive - false_positive))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared distance between the quoted probability and the outcome."""
    return float(np.mean((np.asarray(y_prob, dtype=float) - np.asarray(y_true, dtype=float)) ** 2))


def discrimination(y_true, y_score) -> dict[str, float]:
    """Area under the curve, Gini and Kolmogorov-Smirnov, in one call."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    auc = float(roc_auc_score(y_true, y_score))
    return {"auc": auc, "gini": gini(auc), "ks": ks_statistic(y_true, y_score)}


def calibration_table(y_true, y_prob, n_groups: int = 10) -> pd.DataFrame:
    """Observed against predicted default rate, by group of predicted risk.

    Groups are cut on quantiles of the predicted probability, so each holds
    roughly the same number of loans.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true, dtype=float), "p": np.asarray(y_prob, dtype=float)}
    )
    # duplicates="drop" guards against ties piling up on a quantile boundary.
    frame["group"] = pd.qcut(frame["p"].rank(method="first"), n_groups, labels=False)

    table = frame.groupby("group").agg(
        n=("y", "size"),
        defaults=("y", "sum"),
        predicted=("p", "mean"),
        p_min=("p", "min"),
        p_max=("p", "max"),
    )
    table["observed"] = table["defaults"] / table["n"]
    table["difference"] = table["observed"] - table["predicted"]
    return table.reset_index()


def hosmer_lemeshow(y_true, y_prob, n_groups: int = 10) -> dict[str, float]:
    """Hosmer-Lemeshow goodness-of-fit statistic.

    A large statistic, equivalently a small p-value, says the quoted
    probabilities depart from the observed rates by more than sampling noise
    would explain. On a sample of several hundred thousand loans the test
    rejects on differences far too small to matter in practice, so the decile
    table should be read alongside it rather than the p-value alone.
    """
    table = calibration_table(y_true, y_prob, n_groups=n_groups)
    expected = table["predicted"] * table["n"]
    observed = table["defaults"]
    denominator = expected * (1.0 - table["predicted"])
    denominator = denominator.replace(0, np.nan)

    statistic = float(np.nansum((observed - expected) ** 2 / denominator))
    degrees = max(n_groups - 2, 1)
    p_value = float(stats.chi2.sf(statistic, degrees))
    return {"statistic": statistic, "df": degrees, "p_value": p_value}


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, pd.DataFrame]:
    """Population stability index between a reference and a later sample.

    Bin edges come from the reference sample, as they must: the question is
    whether the later population falls differently into the classes the model
    was built on.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 3:
        raise ValueError("reference sample has too few distinct values to bin")
    edges[0], edges[-1] = -np.inf, np.inf

    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)

    # A share of zero would send the logarithm to infinity; nudge it instead.
    floor = 1e-6
    expected_share = np.maximum(expected_counts / expected_counts.sum(), floor)
    actual_share = np.maximum(actual_counts / actual_counts.sum(), floor)

    contribution = (actual_share - expected_share) * np.log(actual_share / expected_share)
    table = pd.DataFrame(
        {
            "bin": range(len(expected_share)),
            "lower": edges[:-1],
            "upper": edges[1:],
            "expected_share": expected_share,
            "actual_share": actual_share,
            "contribution": contribution,
        }
    )
    return float(contribution.sum()), table


def psi_verdict(psi: float) -> str:
    """Conventional reading of a population stability index."""
    if psi < PSI_MINOR:
        return "stable"
    if psi < PSI_MAJOR:
        return "moderate shift"
    return "major shift, recalibration indicated"


def summarise(y_true, y_prob, label: str = "") -> dict[str, float | str]:
    """Every headline measure for one sample."""
    measures: dict[str, float | str] = {"sample": label, "n": int(len(y_true))}
    measures["default_rate"] = float(np.mean(y_true))
    measures.update(discrimination(y_true, y_prob))
    measures["brier"] = brier_score(y_true, y_prob)
    test = hosmer_lemeshow(y_true, y_prob)
    measures["hl_statistic"] = test["statistic"]
    measures["hl_p_value"] = test["p_value"]
    measures["mean_predicted"] = float(np.mean(y_prob))
    return measures
