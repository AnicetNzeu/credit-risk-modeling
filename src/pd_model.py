"""Probability-of-default model and credit scorecard.

The model is a logistic regression on weight-of-evidence transformed
predictors, which is the form supervisors and model validators expect for a
retail scorecard: every coefficient has a sign that can be argued from credit
reasoning, and the fitted model converts into a points table that a credit
officer can apply by hand.

Sampling follows the two splits a validator asks for.

    Development sample   loans issued 2007 to 2014, split 80/20 at random into
                         an estimation and an out-of-sample part.
    Out-of-time sample   loans issued in 2015, never seen during estimation.

The out-of-time sample matters here more than usual. The observed default rate
climbs from 15.1 per cent in the 2007-2011 book to 19.1 per cent in the 2015
book, so a model fitted on the earlier years meets a genuinely different
population, which is the situation a scorecard faces in production.

Usage
-----
    python src/pd_model.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools.tools import add_constant

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tables as tbl  # noqa: E402
import validation as val  # noqa: E402
import woe  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"
ARTEFACTS = ROOT / "data" / "processed"

RANDOM_STATE = 20261023
DEVELOPMENT_YEARS = range(2007, 2015)   # 2007 to 2014 inclusive
OUT_OF_TIME_YEARS = [2015]

# Information value is a univariate screen, so the floor is deliberately
# permissive: it removes predictors that separate nothing on their own, and
# leaves the real filtering to the multivariate significance test below. The
# conventional 0.02 would discard predictors that are weak alone yet add once
# the others are controlled for.
IV_FLOOR = 0.01
SIGNIFICANCE = 0.01      # threshold for backward elimination
VIF_CEILING = 10.0       # above this, collinearity is usually called severe

# Scorecard calibration, in the usual industry parameterisation.
BASE_SCORE = 600         # score awarded at the base odds
BASE_ODDS = 50.0         # 50 good loans for 1 bad at the base score
PDO = 20                 # points needed to double the odds


def load() -> pd.DataFrame:
    """Read the modelling table produced by prepare_data.py."""
    parquet, csv = PROCESSED / "loans.parquet", PROCESSED / "loans.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv, low_memory=False)
    raise FileNotFoundError("no modelling table; run src/prepare_data.py first")


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Estimation, out-of-sample and out-of-time parts."""
    development = data[data["issue_year"].isin(DEVELOPMENT_YEARS)].copy()
    out_of_time = data[data["issue_year"].isin(OUT_OF_TIME_YEARS)].copy()

    rng = np.random.default_rng(RANDOM_STATE)
    is_estimation = rng.random(len(development)) < 0.80
    estimation = development[is_estimation].copy()
    out_of_sample = development[~is_estimation].copy()
    return estimation, out_of_sample, out_of_time


def fit_logit(
    design: pd.DataFrame,
    target: pd.Series,
    significance: float = SIGNIFICANCE,
) -> object:
    """Fit a logistic regression, then drop insignificant predictors one by one.

    Elimination is backward and removes a single predictor per round, the one
    with the largest p-value, because dropping several at once can remove a
    predictor that would have become significant on its own.
    """
    columns = list(design.columns)
    while True:
        matrix = add_constant(design[columns], has_constant="add")
        model = Logit(target, matrix).fit(disp=0, maxiter=200)

        p_values = model.pvalues.drop("const", errors="ignore")
        worst = p_values.idxmax()
        if p_values[worst] <= significance or len(columns) == 1:
            return model
        print(f"    dropping {worst} (p = {p_values[worst]:.3f})")
        columns.remove(worst)


def predict(model, frame: pd.DataFrame, binnings: dict) -> np.ndarray:
    """Predicted default probability for a new sample."""
    transformed = woe.transform(frame, binnings)
    columns = [c for c in model.params.index if c != "const"]
    matrix = add_constant(transformed[columns], has_constant="add")
    return model.predict(matrix).to_numpy()


def build_scorecard(model, binnings: dict) -> pd.DataFrame:
    """Convert the fitted model into a points table.

    With ``factor = PDO / ln(2)`` and ``offset = BASE_SCORE - factor * ln(BASE_ODDS)``
    the points attached to one class of one predictor are

        points = -(beta * WoE + intercept / k) * factor + offset / k

    where ``k`` is the number of predictors retained. Summing the points of a
    borrower's classes reproduces the model's score, so the table is the model
    rather than an approximation of it.
    """
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)

    retained = [c for c in model.params.index if c != "const"]
    k = len(retained)
    intercept = float(model.params["const"])

    rows = []
    for column in retained:
        name = column.removeprefix("woe_")
        beta = float(model.params[column])
        table = binnings[name].table
        for _, entry in table.iterrows():
            points = -(beta * entry["woe"] + intercept / k) * factor + offset / k
            rows.append(
                {
                    "predictor": name,
                    "class": entry["class"],
                    "share": entry["share"],
                    "event_rate": entry["event_rate"],
                    "woe": entry["woe"],
                    "coefficient": beta,
                    "points": round(points),
                }
            )

    scorecard = pd.DataFrame(rows)
    scorecard["points"] = scorecard["points"].astype(int)
    return scorecard


def variance_inflation(design: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor of every column of the design matrix.

    VIF_j is 1 / (1 - R_j^2) where R_j^2 comes from regressing predictor j on
    all the others. A value above ten is the usual signal that a coefficient
    is being estimated on very little independent variation, which makes its
    sign unstable and its interpretation unsafe.
    """
    matrix = design.to_numpy(dtype=float)
    rows = []
    for index, column in enumerate(design.columns):
        others = np.delete(matrix, index, axis=1)
        others = np.column_stack([np.ones(len(others)), others])
        target = matrix[:, index]
        coefficients, *_ = np.linalg.lstsq(others, target, rcond=None)
        residual = target - others @ coefficients
        total_ss = float(np.sum((target - target.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / total_ss if total_ss > 0 else 0.0
        vif = 1.0 / (1.0 - r_squared) if r_squared < 1 else np.inf
        rows.append(
            {
                "predictor": column.removeprefix("woe_"),
                "r_squared": r_squared,
                "vif": vif,
                "verdict": "severe" if vif > VIF_CEILING else "acceptable",
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def recalibrate_intercept(
    model, frame: pd.DataFrame, binnings: dict, target: pd.Series
) -> float:
    """Shift the intercept so the model reproduces an observed default rate.

    Discrimination and calibration are separate properties. When a model still
    ranks borrowers correctly on a later book but quotes the wrong level, the
    remedy is not to refit it but to move the intercept, which leaves the
    ranking untouched. The shift is found by maximising the likelihood over
    the intercept alone, holding every slope fixed.
    """
    transformed = woe.transform(frame, binnings)
    columns = [c for c in model.params.index if c != "const"]
    # Start from the fitted linear predictor, intercept included, so that the
    # quantity returned is a genuine shift and not the new intercept.
    linear = (
        float(model.params["const"])
        + transformed[columns].to_numpy() @ model.params[columns].to_numpy()
    )
    y = target.to_numpy(dtype=float)

    # One-dimensional Newton step, iterated to convergence.
    shift = 0.0
    for _ in range(100):
        probability = 1.0 / (1.0 + np.exp(-(linear + shift)))
        gradient = float(np.sum(y - probability))
        hessian = float(np.sum(probability * (1.0 - probability)))
        if hessian <= 0:
            break
        step = gradient / hessian
        shift += step
        if abs(step) < 1e-10:
            break
    return shift


def apply_scorecard(
    frame: pd.DataFrame, scorecard: pd.DataFrame, binnings: dict
) -> np.ndarray:
    """Score a sample by summing the points of each borrower's classes."""
    total = np.zeros(len(frame), dtype=float)
    for predictor, block in scorecard.groupby("predictor"):
        labels = woe.assign_classes(frame[predictor], binnings[predictor])
        lookup = block.set_index("class")["points"]
        total += labels.map(lookup).astype(float).fillna(0.0).to_numpy()
    return total


def verify_scorecard(
    frame: pd.DataFrame,
    scorecard: pd.DataFrame,
    binnings: dict,
    probability: np.ndarray,
) -> dict[str, float]:
    """Check that the points table reproduces the model it was derived from.

    The two should agree up to the rounding of each attribute's points to an
    integer, so the largest gap is expected to be a few points at most. A
    larger gap means the table and the model have parted company, which is a
    defect a validator would be right to reject.
    """
    from_points = apply_scorecard(frame, scorecard, binnings)
    from_model = score_from_probability(probability)
    difference = from_points - from_model
    return {
        "max_absolute_gap": float(np.max(np.abs(difference))),
        "mean_gap": float(np.mean(difference)),
        "predictors": int(scorecard["predictor"].nunique()),
    }


def score_from_probability(probability: np.ndarray) -> np.ndarray:
    """Translate a probability into the scorecard's integer scale."""
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-9, 1 - 1e-9)
    odds_good = (1.0 - probability) / probability
    return offset + factor * np.log(odds_good)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    data = load()
    estimation, out_of_sample, out_of_time = split(data)

    print("Samples")
    for name, frame in (
        ("estimation (2007-2014)", estimation),
        ("out of sample (2007-2014)", out_of_sample),
        ("out of time (2015)", out_of_time),
    ):
        rate = frame["default"].mean()
        print(f"  {name:<28} n = {len(frame):>7,}".replace(",", " ") + f"   default rate = {rate:.2%}")

    target = estimation["default"]

    # --- binning and screening -------------------------------------------
    print("\nBinning predictors on the estimation sample")
    from prepare_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    binnings = woe.fit_all(
        estimation, target, NUMERIC_FEATURES, CATEGORICAL_FEATURES, max_bins=8
    )
    iv_table = woe.information_values(binnings)
    print(iv_table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(iv_table, TABLES / "information_values.md")
    iv_table.to_csv(TABLES / "information_values.csv", index=False)

    retained = iv_table.loc[iv_table["iv"] >= IV_FLOOR, "predictor"].tolist()
    dropped = sorted(set(binnings) - set(retained))
    print(f"\n  retained {len(retained)} predictors with IV >= {IV_FLOOR}")
    if dropped:
        print(f"  dropped {len(dropped)}: {', '.join(dropped)}")
    binnings = {name: binnings[name] for name in retained}

    # --- estimation -------------------------------------------------------
    print("\nFitting the logistic regression")
    design = woe.transform(estimation, binnings)
    model = fit_logit(design, target)
    print(f"  {len(model.params) - 1} predictors retained after elimination")
    print(f"  pseudo R-squared (McFadden) = {model.prsquared:.4f}")

    coefficients = pd.DataFrame(
        {
            "predictor": [c.removeprefix("woe_") for c in model.params.index],
            "coefficient": model.params.to_numpy(),
            "std_error": model.bse.to_numpy(),
            "z": model.tvalues.to_numpy(),
            "p_value": model.pvalues.to_numpy(),
        }
    )
    tbl.write(coefficients, TABLES / "pd_coefficients.md")

    retained_columns = [c for c in model.params.index if c != "const"]
    vif_table = variance_inflation(design[retained_columns])
    severe = vif_table.loc[vif_table["vif"] > VIF_CEILING, "predictor"].tolist()
    print(f"  largest variance inflation factor = {vif_table['vif'].max():.2f}"
          + (f"   severe: {', '.join(severe)}" if severe else "   none severe"))
    tbl.write(vif_table, TABLES / "pd_variance_inflation.md")

    # --- validation -------------------------------------------------------
    print("\nValidation")
    samples = {
        "estimation": estimation,
        "out of sample": out_of_sample,
        "out of time": out_of_time,
    }
    predictions = {name: predict(model, frame, binnings) for name, frame in samples.items()}

    rows = [
        val.summarise(frame["default"].to_numpy(), predictions[name], label=name)
        for name, frame in samples.items()
    ]
    performance = pd.DataFrame(rows)
    print(performance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(performance, TABLES / "pd_performance.md")

    for name, frame in samples.items():
        table = val.calibration_table(frame["default"].to_numpy(), predictions[name])
        tbl.write(table, TABLES / f"calibration_{name.replace(' ', '_')}.md")

    # --- stability --------------------------------------------------------
    psi, psi_table = val.population_stability_index(
        predictions["estimation"], predictions["out of time"]
    )
    print(f"\nPopulation stability index, estimation against out of time: {psi:.4f}"
          f"  ({val.psi_verdict(psi)})")
    tbl.write(psi_table, TABLES / "psi_estimation_vs_out_of_time.md")

    # --- recalibration ----------------------------------------------------
    # The population barely moves, yet the later book defaults more often, so
    # the gap is in the level rather than in the ranking.
    observed_oot = float(out_of_time["default"].mean())
    predicted_oot = float(np.mean(predictions["out of time"]))
    print(f"\nOut of time: observed default rate {observed_oot:.2%},"
          f" quoted {predicted_oot:.2%}, shortfall {observed_oot - predicted_oot:.2%}")

    shift = recalibrate_intercept(model, out_of_time, binnings, out_of_time["default"])
    transformed_oot = woe.transform(out_of_time, binnings)
    columns_oot = [c for c in model.params.index if c != "const"]
    linear_oot = (
        float(model.params["const"])
        + transformed_oot[columns_oot].to_numpy() @ model.params[columns_oot].to_numpy()
    )
    recalibrated = 1.0 / (1.0 + np.exp(-(linear_oot + shift)))

    before = val.summarise(out_of_time["default"].to_numpy(), predictions["out of time"],
                           label="out of time, as fitted")
    after = val.summarise(out_of_time["default"].to_numpy(), recalibrated,
                          label="out of time, intercept recalibrated")
    recalibration = pd.DataFrame([before, after])
    print(f"  intercept moves from {float(model.params['const']):.4f}"
          f" to {float(model.params['const']) + shift:.4f} (shift {shift:+.4f})")
    print(recalibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(recalibration, TABLES / "pd_recalibration.md")

    # --- benchmark including the platform's own assessment ----------------
    print("\nBenchmark including LendingClub's grade and interest rate")
    from prepare_data import LENDER_ASSESSMENT

    bench_numeric = NUMERIC_FEATURES + ["int_rate", "installment"]
    bench_categorical = CATEGORICAL_FEATURES + ["grade", "sub_grade"]
    bench_binnings = woe.fit_all(
        estimation, target, bench_numeric, bench_categorical, max_bins=8
    )
    bench_iv = woe.information_values(bench_binnings)
    bench_retained = bench_iv.loc[bench_iv["iv"] >= IV_FLOOR, "predictor"].tolist()
    bench_binnings = {n: bench_binnings[n] for n in bench_retained}
    bench_model = fit_logit(woe.transform(estimation, bench_binnings), target)

    bench_rows = []
    for name, frame in samples.items():
        probability = predict(bench_model, frame, bench_binnings)
        bench_rows.append(val.summarise(frame["default"].to_numpy(), probability, label=name))
    bench_performance = pd.DataFrame(bench_rows)
    print(bench_performance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(bench_performance, TABLES / "pd_performance_benchmark.md")

    comparison = pd.DataFrame(
        {
            "sample": performance["sample"],
            "auc_borrower_only": performance["auc"],
            "auc_with_lender_grade": bench_performance["auc"],
            "gini_borrower_only": performance["gini"],
            "gini_with_lender_grade": bench_performance["gini"],
        }
    )
    tbl.write(comparison, TABLES / "pd_model_comparison.md")

    # --- scorecard --------------------------------------------------------
    print("\nBuilding the scorecard")
    scorecard = build_scorecard(model, binnings)
    tbl.write(scorecard, TABLES / "scorecard.md")
    scorecard.to_csv(TABLES / "scorecard.csv", index=False)
    print(f"  {len(scorecard)} rows, {scorecard['predictor'].nunique()} predictors")
    print(f"  points range from {scorecard['points'].min()} to {scorecard['points'].max()}")

    check = verify_scorecard(
        out_of_sample, scorecard, binnings, predictions["out of sample"]
    )
    print(f"  reproduction check: largest gap with the model's own score ="
          f" {check['max_absolute_gap']:.2f} points")
    total_points = apply_scorecard(out_of_sample, scorecard, binnings)
    print(f"  scores on the out-of-sample part span"
          f" {total_points.min():.0f} to {total_points.max():.0f}")

    # --- store predictions for the loss and exposure stages ---------------
    for name, frame in samples.items():
        frame["pd_predicted"] = predictions[name]
        frame["score"] = score_from_probability(predictions[name])
        frame["sample"] = name
    combined = pd.concat(samples.values(), ignore_index=True)
    combined.to_csv(ARTEFACTS / "loans_with_pd.csv", index=False)

    summary = {
        "n_total": int(len(combined)),
        "default_rate": float(combined["default"].mean()),
        "predictors_retained": int(len(model.params) - 1),
        "mcfadden_r2": float(model.prsquared),
        "psi_estimation_vs_out_of_time": float(psi),
        "performance": performance.to_dict(orient="records"),
        "performance_benchmark": bench_performance.to_dict(orient="records"),
    }
    (TABLES / "pd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTables written to {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
