"""Weight-of-evidence binning and information-value screening.

Credit scorecards are normally built on binned predictors rather than on raw
values. Every predictor is cut into a small number of classes, and each class
is replaced by its weight of evidence

    WoE_k = ln( share of non-defaults in class k / share of defaults in class k )

The logistic regression is then fitted on these transformed values. The
construction handles missing values as a class of their own, bounds the
influence of outliers, and linearises the relationship between a predictor and
the log-odds of default, which is what the logit model assumes.

Information value summarises how well a binned predictor separates the two
outcomes,

    IV = sum_k ( share of non-defaults - share of defaults )_k * WoE_k

and is used here only as a first screen. The convention followed in the
industry, and applied in this project, is to drop predictors below 0.02, treat
0.02 to 0.1 as weak, 0.1 to 0.3 as medium and 0.3 to 0.5 as strong, and to
inspect anything above 0.5 for leakage rather than celebrate it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

MISSING_LABEL = "Missing"

# Laplace-style correction applied to the class counts so that a class holding
# no default, or no non-default, still yields a finite weight of evidence.
SMOOTHING = 0.5


@dataclass
class Binning:
    """Binning of a single predictor, with its weight-of-evidence table.

    Attributes
    ----------
    name:
        Predictor the binning applies to.
    kind:
        Either ``"numeric"`` or ``"categorical"``.
    table:
        One row per class, carrying the counts, the event rate, the weight of
        evidence and the information-value contribution.
    edges:
        Cut points, for numeric predictors only.
    mapping:
        Raw category to class label, for categorical predictors only.
    iv:
        Information value, the sum of the per-class contributions.
    """

    name: str
    kind: str
    table: pd.DataFrame
    edges: np.ndarray | None = None
    mapping: dict[object, str] = field(default_factory=dict)
    iv: float = 0.0

    def transform(self, values: pd.Series) -> pd.Series:
        """Replace raw values by the weight of evidence of their class."""
        labels = assign_classes(values, self)
        lookup = self.table.set_index("class")["woe"]
        # A category never seen in training carries no evidence either way.
        return labels.map(lookup).astype(float).fillna(0.0)


def _discrete_label(value: object) -> str:
    """Render a numeric value as a stable class key.

    Counts arrive as floats, so 2.0 has to become "2" and not "2.0", or the
    label will not match the one recorded at fitting time.
    """
    if pd.isna(value):
        return MISSING_LABEL
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return str(int(number)) if number.is_integer() else f"{number:g}"


def assign_classes(values: pd.Series, binning: Binning) -> pd.Series:
    """Map raw values onto the class labels of a fitted binning."""
    if binning.kind == "discrete":
        keys = pd.to_numeric(values, errors="coerce").map(_discrete_label)
        return keys.map(lambda k: binning.mapping.get(k, "Other"))

    if binning.kind == "numeric":
        numeric = pd.to_numeric(values, errors="coerce")
        labels = pd.cut(numeric, bins=binning.edges, include_lowest=True)
        labels = labels.astype(object)
        labels = labels.where(~pd.isna(labels), MISSING_LABEL)
        return pd.Series([str(v) for v in labels], index=values.index)

    as_text = values.astype(object).where(~pd.isna(values), MISSING_LABEL)
    as_text = as_text.astype(str)
    return as_text.map(lambda v: binning.mapping.get(v, MISSING_LABEL))


def _woe_table(labels: pd.Series, target: pd.Series) -> tuple[pd.DataFrame, float]:
    """Build the weight-of-evidence table for one set of class labels.

    ``target`` is the default flag, 1 for a default and 0 otherwise.
    """
    frame = pd.DataFrame({"class": labels.values, "target": target.values})
    grouped = frame.groupby("class", dropna=False)["target"].agg(["count", "sum"])
    grouped.columns = ["n", "defaults"]
    grouped["non_defaults"] = grouped["n"] - grouped["defaults"]

    total_defaults = float(grouped["defaults"].sum())
    total_non_defaults = float(grouped["non_defaults"].sum())
    if total_defaults == 0 or total_non_defaults == 0:
        raise ValueError("the sample contains a single outcome; WoE is undefined")

    share_default = (grouped["defaults"] + SMOOTHING) / (total_defaults + SMOOTHING * len(grouped))
    share_non_default = (grouped["non_defaults"] + SMOOTHING) / (
        total_non_defaults + SMOOTHING * len(grouped)
    )

    grouped["event_rate"] = grouped["defaults"] / grouped["n"]
    grouped["share"] = grouped["n"] / grouped["n"].sum()
    grouped["woe"] = np.log(share_non_default / share_default)
    grouped["iv_part"] = (share_non_default - share_default) * grouped["woe"]

    table = grouped.reset_index()
    return table, float(table["iv_part"].sum())


def fit_numeric(
    values: pd.Series,
    target: pd.Series,
    name: str,
    max_bins: int = 8,
    min_share: float = 0.05,
) -> Binning:
    """Bin a numeric predictor on quantiles, then merge thin classes.

    Quantile cuts keep every class populated, which matters for the stability
    of the weights. Classes holding less than ``min_share`` of the sample are
    merged into their neighbour, since a weight estimated on a handful of
    observations will not survive out of sample.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    observed = numeric.dropna()
    if observed.empty:
        raise ValueError(f"{name}: no usable numeric value")

    quantiles = np.linspace(0, 1, max_bins + 1)
    edges = np.unique(np.nanquantile(observed, quantiles))
    if len(edges) < 3:
        raise ValueError(f"{name}: not enough distinct values to bin")

    # Open the outer edges so that unseen extremes still fall inside a class.
    edges[0] = -np.inf
    edges[-1] = np.inf

    labels = pd.cut(numeric, bins=edges, include_lowest=True).astype(object)
    labels = pd.Series(
        [MISSING_LABEL if pd.isna(v) else str(v) for v in labels], index=values.index
    )

    # Merge any class that is too thin to support a stable weight.
    while True:
        shares = labels.value_counts(normalize=True)
        thin = [c for c in shares.index if c != MISSING_LABEL and shares[c] < min_share]
        if not thin or len(edges) <= 3:
            break
        interior = edges[1:-1]
        if len(interior) == 0:
            break
        # Drop the cut point adjacent to the thinnest class.
        thinnest = min(thin, key=lambda c: shares[c])
        ordered = [c for c in sorted(shares.index) if c != MISSING_LABEL]
        position = min(ordered.index(thinnest), len(interior) - 1)
        edges = np.delete(edges, position + 1)
        labels = pd.cut(numeric, bins=edges, include_lowest=True).astype(object)
        labels = pd.Series(
            [MISSING_LABEL if pd.isna(v) else str(v) for v in labels], index=values.index
        )

    table, iv = _woe_table(labels, target)
    return Binning(name=name, kind="numeric", table=table, edges=edges, iv=iv)


def fit_discrete(
    values: pd.Series,
    target: pd.Series,
    name: str,
    min_share: float = 0.01,
) -> Binning:
    """Bin a numeric predictor that takes only a handful of values.

    Counts of past delinquencies, public records or recent enquiries are zero
    for most borrowers. Cutting such a variable on quantiles produces classes
    that all hold the same value and therefore separate nothing, which makes a
    genuinely predictive variable look useless. Here each observed value forms
    its own class and the thin tail is pooled into one.
    """
    keys = pd.to_numeric(values, errors="coerce").map(_discrete_label)
    shares = keys.value_counts(normalize=True)
    keep = set(shares[shares >= min_share].index)

    mapping = {level: (level if level in keep else "Other") for level in shares.index}
    labels = keys.map(mapping)

    table, iv = _woe_table(labels, target)
    return Binning(name=name, kind="discrete", table=table, mapping=mapping, iv=iv)


def fit_categorical(
    values: pd.Series,
    target: pd.Series,
    name: str,
    min_share: float = 0.02,
) -> Binning:
    """Bin a categorical predictor, pooling rare levels into one class.

    Levels thinner than ``min_share`` are pooled into ``"Other"`` rather than
    carrying a weight of their own, for the same stability reason.
    """
    as_text = values.astype(object).where(~pd.isna(values), MISSING_LABEL).astype(str)
    shares = as_text.value_counts(normalize=True)
    keep = set(shares[shares >= min_share].index)

    mapping = {level: (level if level in keep else "Other") for level in shares.index}
    labels = as_text.map(mapping)

    table, iv = _woe_table(labels, target)
    return Binning(name=name, kind="categorical", table=table, mapping=mapping, iv=iv)


def fit_all(
    frame: pd.DataFrame,
    target: pd.Series,
    numeric: list[str],
    categorical: list[str],
    max_bins: int = 8,
    discrete_threshold: int = 15,
) -> dict[str, Binning]:
    """Bin every listed predictor, skipping those that cannot be binned.

    A numeric predictor with at most ``discrete_threshold`` distinct values is
    treated as discrete rather than cut on quantiles.
    """
    binnings: dict[str, Binning] = {}

    for column in numeric:
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        if series.notna().sum() == 0:
            continue
        try:
            if series.nunique(dropna=True) <= discrete_threshold:
                binnings[column] = fit_discrete(series, target, column)
            else:
                binnings[column] = fit_numeric(series, target, column, max_bins=max_bins)
        except ValueError:
            continue

    for column in categorical:
        if column not in frame.columns:
            continue
        try:
            binnings[column] = fit_categorical(frame[column], target, column)
        except ValueError:
            continue

    return binnings


def information_values(binnings: dict[str, Binning]) -> pd.DataFrame:
    """Rank predictors by information value, with the usual reading."""

    def strength(iv: float) -> str:
        if iv < 0.02:
            return "unpredictive"
        if iv < 0.1:
            return "weak"
        if iv < 0.3:
            return "medium"
        if iv < 0.5:
            return "strong"
        return "suspiciously strong"

    rows = [
        {"predictor": name, "iv": b.iv, "classes": len(b.table), "strength": strength(b.iv)}
        for name, b in binnings.items()
    ]
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def transform(frame: pd.DataFrame, binnings: dict[str, Binning]) -> pd.DataFrame:
    """Apply a set of fitted binnings, returning one WoE column per predictor."""
    return pd.DataFrame(
        {f"woe_{name}": b.transform(frame[name]) for name, b in binnings.items()},
        index=frame.index,
    )
