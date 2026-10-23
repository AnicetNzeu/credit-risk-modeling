"""Loss given default, exposure at default, and the expected credit loss.

Both quantities are observed only on loans that actually defaulted, so both
models are estimated on the defaulted part of the sample and then applied to
the whole portfolio. The samples are the ones the probability-of-default stage
uses, so a loan that was out of time there is out of time here too.

Loss given default
    The recovery rate is zero for about one defaulted loan in five and
    positive but small for the rest, so a single regression would fit neither
    part well. The model is therefore in two stages, the usual treatment for a
    variable with a mass point at zero:

        E[recovery rate] = P(recovery > 0) x E[recovery rate | recovery > 0]
        LGD              = 1 - E[recovery rate]

    The first stage is a logistic regression on weight-of-evidence predictors,
    as in the probability-of-default model. The second is a linear regression
    among the loans that recovered something. Weight of evidence is defined
    against a binary outcome and so cannot carry over to the second stage;
    there the binning classes enter as indicator variables instead, with the
    largest class held out as the reference.

Exposure at default
    The credit conversion factor is the share of the granted amount still owed
    when the loan defaults,

        CCF = (amount granted - principal repaid) / amount granted
        EAD = CCF x amount granted

    It is bounded between zero and one and is modelled by linear regression on
    the same indicator variables, with predictions clipped back into range.

Expected credit loss
    ECL = PD x LGD x EAD, computed loan by loan and then aggregated. This is
    the twelve-month expected loss of IFRS 9 stage one, in the sense that the
    probability of default is estimated over the loan's horizon rather than
    over its whole life.

Usage
-----
    python src/lgd_ead_model.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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

IV_FLOOR = 0.01


def load() -> pd.DataFrame:
    """Read the table the probability-of-default stage wrote."""
    path = PROCESSED / "loans_with_pd.csv"
    if not path.exists():
        raise FileNotFoundError("run src/pd_model.py first")
    return pd.read_csv(path, low_memory=False)


def dummies(frame: pd.DataFrame, binnings: dict) -> pd.DataFrame:
    """One indicator per binning class, the largest class held out.

    Dropping the largest rather than the first class keeps the reference
    group well populated, so the remaining coefficients are read against a
    stable baseline.
    """
    blocks = []
    for name, binning in binnings.items():
        labels = woe.assign_classes(frame[name], binning)
        encoded = pd.get_dummies(labels, prefix=name, dtype=float)
        reference = binning.table.loc[binning.table["n"].idxmax(), "class"]
        encoded = encoded.drop(columns=[f"{name}_{reference}"], errors="ignore")
        blocks.append(encoded)
    out = pd.concat(blocks, axis=1)
    out.index = frame.index
    return out


def align(design: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Give a new sample exactly the columns the model was fitted on."""
    return design.reindex(columns=columns, fill_value=0.0)


def regression_report(y_true, y_pred, label: str) -> dict[str, float | str]:
    """Fit quality of a continuous prediction, against a mean benchmark."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    total = y_true - y_true.mean()
    r_squared = 1.0 - float(residual @ residual) / float(total @ total)
    return {
        "sample": label,
        "n": int(len(y_true)),
        "observed_mean": float(y_true.mean()),
        "predicted_mean": float(y_pred.mean()),
        "r_squared": r_squared,
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "correlation": float(np.corrcoef(y_true, y_pred)[0, 1]),
    }


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    data = load()

    from prepare_data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    defaults = data[data["default"] == 1].copy()
    estimation = defaults[defaults["sample"] == "estimation"].copy()
    samples = {
        name: defaults[defaults["sample"] == name].copy()
        for name in ("estimation", "out of sample", "out of time")
    }

    print("Defaulted loans available for the loss and exposure models")
    for name, frame in samples.items():
        print(f"  {name:<16} n = {len(frame):>7,}".replace(",", " "))

    # =====================================================================
    # Loss given default, first stage: is anything recovered at all?
    # =====================================================================
    print("\nLoss given default, first stage: probability of a positive recovery")
    estimation["recovered"] = (estimation["recovery_rate"] > 0).astype(int)
    for frame in samples.values():
        frame["recovered"] = (frame["recovery_rate"] > 0).astype(int)

    binnings = woe.fit_all(
        estimation, estimation["recovered"], NUMERIC_FEATURES, CATEGORICAL_FEATURES
    )
    iv_table = woe.information_values(binnings)
    binnings = {
        n: b for n, b in binnings.items()
        if n in set(iv_table.loc[iv_table["iv"] >= IV_FLOOR, "predictor"])
    }
    tbl.write(iv_table, TABLES / "lgd_stage1_information_values.md")
    print(f"  {len(binnings)} predictors retained, "
          f"largest information value {iv_table['iv'].max():.4f}")

    design = woe.transform(estimation, binnings)
    stage1 = Logit(estimation["recovered"], add_constant(design, has_constant="add")).fit(
        disp=0, maxiter=200
    )
    stage1_columns = [c for c in stage1.params.index if c != "const"]

    def stage1_predict(frame: pd.DataFrame) -> np.ndarray:
        matrix = add_constant(
            woe.transform(frame, binnings)[stage1_columns], has_constant="add"
        )
        return stage1.predict(matrix).to_numpy()

    stage1_rows = [
        val.summarise(frame["recovered"].to_numpy(), stage1_predict(frame), label=name)
        for name, frame in samples.items()
    ]
    stage1_performance = pd.DataFrame(stage1_rows)
    print(stage1_performance[["sample", "n", "auc", "gini", "brier"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(stage1_performance, TABLES / "lgd_stage1_performance.md")

    # =====================================================================
    # Loss given default, second stage: how much, when something is recovered
    # =====================================================================
    print("\nLoss given default, second stage: size of the recovery")
    recovered = estimation[estimation["recovered"] == 1]
    stage2_design = dummies(recovered, binnings)
    stage2_columns = list(stage2_design.columns)
    stage2 = LinearRegression().fit(stage2_design.to_numpy(), recovered["recovery_rate"].to_numpy())

    def stage2_predict(frame: pd.DataFrame) -> np.ndarray:
        matrix = align(dummies(frame, binnings), stage2_columns).to_numpy()
        return np.clip(stage2.predict(matrix), 0.0, 1.0)

    stage2_rows = []
    for name, frame in samples.items():
        subset = frame[frame["recovered"] == 1]
        stage2_rows.append(
            regression_report(subset["recovery_rate"], stage2_predict(subset), name)
        )
    stage2_performance = pd.DataFrame(stage2_rows)
    print(stage2_performance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(stage2_performance, TABLES / "lgd_stage2_performance.md")

    # =====================================================================
    # Loss given default, the two stages combined
    # =====================================================================
    print("\nLoss given default, combined")
    lgd_rows = []
    for name, frame in samples.items():
        expected_recovery = stage1_predict(frame) * stage2_predict(frame)
        predicted_lgd = np.clip(1.0 - expected_recovery, 0.0, 1.0)
        frame["lgd_predicted"] = predicted_lgd
        lgd_rows.append(
            regression_report(frame["lgd_observed"], predicted_lgd, name)
        )
    lgd_performance = pd.DataFrame(lgd_rows)
    print(lgd_performance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(lgd_performance, TABLES / "lgd_performance.md")

    # The 2015 book recovers more than the earlier ones, which the model does
    # not anticipate. Its ranking of loans still holds, so the remedy is the
    # same as for the probability of default: move the level, keep the slopes.
    out_of_time = samples["out of time"]
    bias = float(out_of_time["lgd_observed"].mean() - out_of_time["lgd_predicted"].mean())
    shifted = np.clip(out_of_time["lgd_predicted"] + bias, 0.0, 1.0)
    recalibration = pd.DataFrame(
        [
            regression_report(
                out_of_time["lgd_observed"], out_of_time["lgd_predicted"],
                "out of time, as fitted",
            ),
            regression_report(
                out_of_time["lgd_observed"], shifted,
                "out of time, level recalibrated",
            ),
        ]
    )
    print(f"\n  level correction applied to the 2015 book: {bias:+.4f}")
    print(recalibration.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(recalibration, TABLES / "lgd_recalibration.md")

    # =====================================================================
    # Exposure at default
    # =====================================================================
    print("\nExposure at default: credit conversion factor")
    ead_binnings = woe.fit_all(
        estimation,
        (estimation["ccf"] > estimation["ccf"].median()).astype(int),
        NUMERIC_FEATURES,
        CATEGORICAL_FEATURES,
    )
    ead_design = dummies(estimation, ead_binnings)
    ead_columns = list(ead_design.columns)
    ead_model = LinearRegression().fit(ead_design.to_numpy(), estimation["ccf"].to_numpy())

    def ccf_predict(frame: pd.DataFrame) -> np.ndarray:
        matrix = align(dummies(frame, ead_binnings), ead_columns).to_numpy()
        return np.clip(ead_model.predict(matrix), 0.0, 1.0)

    ead_rows = []
    for name, frame in samples.items():
        predicted_ccf = ccf_predict(frame)
        frame["ccf_predicted"] = predicted_ccf
        frame["ead_predicted"] = predicted_ccf * frame["funded_amnt"]
        ead_rows.append(regression_report(frame["ccf"], predicted_ccf, name))
    ead_performance = pd.DataFrame(ead_rows)
    print(ead_performance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.write(ead_performance, TABLES / "ead_performance.md")

    ead_amount_rows = [
        regression_report(frame["ead_observed"], frame["ead_predicted"], name)
        for name, frame in samples.items()
    ]
    tbl.write(pd.DataFrame(ead_amount_rows), TABLES / "ead_amount_performance.md", decimals=2)

    # =====================================================================
    # Expected credit loss on the whole portfolio
    # =====================================================================
    print("\nExpected credit loss on the whole portfolio")
    portfolio = data.copy()
    portfolio["lgd_predicted"] = np.clip(
        1.0 - stage1_predict(portfolio) * stage2_predict(portfolio), 0.0, 1.0
    )
    portfolio["ccf_predicted"] = ccf_predict(portfolio)
    portfolio["ead_predicted"] = portfolio["ccf_predicted"] * portfolio["funded_amnt"]
    portfolio["ecl"] = (
        portfolio["pd_predicted"] * portfolio["lgd_predicted"] * portfolio["ead_predicted"]
    )

    # Loss actually borne: principal never repaid, net of what was recovered.
    portfolio["realised_loss"] = np.where(
        portfolio["default"] == 1,
        (portfolio["ead_observed"] - portfolio["recovery_rate"] * portfolio["funded_amnt"]).clip(lower=0),
        0.0,
    )

    ecl_rows = []
    for name in ("estimation", "out of sample", "out of time"):
        block = portfolio[portfolio["sample"] == name]
        exposure = float(block["funded_amnt"].sum())
        expected = float(block["ecl"].sum())
        realised = float(block["realised_loss"].sum())
        ecl_rows.append(
            {
                "sample": name,
                "loans": int(len(block)),
                "exposure_usd": exposure,
                "expected_loss_usd": expected,
                "realised_loss_usd": realised,
                "expected_loss_pct": 100.0 * expected / exposure,
                "realised_loss_pct": 100.0 * realised / exposure,
                "coverage_ratio": expected / realised if realised else np.nan,
            }
        )
    ecl_summary = pd.DataFrame(ecl_rows)
    print(ecl_summary.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    tbl.write(ecl_summary, TABLES / "ecl_summary.md", decimals=2)
    ecl_summary.to_csv(TABLES / "ecl_summary.csv", index=False)

    # Expected loss by risk grade of the scorecard, the view a credit
    # committee actually looks at.
    portfolio["score_band"] = pd.cut(
        portfolio["score"],
        bins=[-np.inf, 500, 520, 540, 560, np.inf],
        labels=["below 500", "500-520", "520-540", "540-560", "above 560"],
    )
    by_band = portfolio.groupby("score_band", observed=True).agg(
        loans=("default", "size"),
        observed_default_rate=("default", "mean"),
        mean_pd=("pd_predicted", "mean"),
        mean_lgd=("lgd_predicted", "mean"),
        mean_ead=("ead_predicted", "mean"),
        exposure_usd=("funded_amnt", "sum"),
        expected_loss_usd=("ecl", "sum"),
        realised_loss_usd=("realised_loss", "sum"),
    )
    by_band["expected_loss_pct"] = 100 * by_band["expected_loss_usd"] / by_band["exposure_usd"]
    by_band["realised_loss_pct"] = 100 * by_band["realised_loss_usd"] / by_band["exposure_usd"]
    by_band = by_band.reset_index()
    print()
    print(by_band.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    tbl.write(by_band, TABLES / "ecl_by_score_band.md", decimals=2)
    by_band.to_csv(TABLES / "ecl_by_score_band.csv", index=False)

    portfolio.to_csv(PROCESSED / "loans_with_ecl.csv", index=False)

    summary = {
        "defaults_modelled": int(len(defaults)),
        "share_with_zero_recovery": float((defaults["recovery_rate"] == 0).mean()),
        "mean_observed_lgd": float(defaults["lgd_observed"].mean()),
        "mean_observed_ccf": float(defaults["ccf"].mean()),
        "lgd_stage1": stage1_performance.to_dict(orient="records"),
        "lgd_combined": lgd_performance.to_dict(orient="records"),
        "ead": ead_performance.to_dict(orient="records"),
        "ecl": ecl_summary.to_dict(orient="records"),
    }
    (TABLES / "lgd_ead_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nTables written to {TABLES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
