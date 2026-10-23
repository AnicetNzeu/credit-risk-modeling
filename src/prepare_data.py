"""Assemble a modelling table from the raw LendingClub archives.

The four archives hold loans issued between 2007 and 2015. Every loan in them
has reached a terminal status, either fully repaid or charged off, so the
default indicator is observed for the whole sample and no censoring correction
is needed.

Three conventions govern this file and are worth stating plainly, because each
is a modelling choice rather than a fact about the data.

Default definition
    A loan counts as defaulted when its status is "Charged Off". LendingClub
    writes a loan off after 120 days past due, which is stricter than the 90
    days of the Basel reference definition; the indicator is therefore mildly
    conservative.

Loans outside current credit policy
    The 2007-2011 archive flags loans that would not be approved under the
    policy in force when the file was published. They are kept, since they are
    genuine observed outcomes, and marked by ``outside_credit_policy`` so the
    choice can be reversed.

Separation of origination and outcome
    Only variables known when the loan is granted may enter the probability of
    default model. Everything recorded as the loan runs its course, payments
    received and amounts recovered above all, is kept aside for the loss and
    exposure models. Mixing the two is the most common way to produce a
    scorecard that looks excellent and predicts nothing.

Usage
-----
    python src/prepare_data.py
"""

from __future__ import annotations

import io
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

VINTAGES = {
    "LoanStats3a": "2007-2011",
    "LoanStats3b": "2012-2013",
    "LoanStats3c": "2014",
    "LoanStats3d": "2015",
}

# Known at origination, and free of LendingClub's own risk assessment.
NUMERIC_FEATURES = [
    "loan_amnt",
    "annual_inc",
    "dti",
    "delinq_2yrs",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "emp_length_years",
    "credit_history_months",
    "pub_rec_bankruptcies",
]

CATEGORICAL_FEATURES = [
    "term_months",
    "home_ownership",
    "verification_status",
    "purpose",
    "addr_state",
]

# LendingClub's own grade and the interest rate it implies. These are known at
# origination too, but they are the output of the platform's internal model,
# so a scorecard built on them largely re-estimates that model. They are kept
# apart and used only as a benchmark.
LENDER_ASSESSMENT = ["grade", "sub_grade", "int_rate", "installment"]

# Recorded only as the loan runs. Never predictors.
OUTCOME_COLUMNS = [
    "loan_status",
    "recoveries",
    "total_rec_prncp",
    "total_pymnt",
    "out_prncp",
    "funded_amnt",
]

RAW_COLUMNS = [
    "loan_amnt",
    "funded_amnt",
    "term",
    "int_rate",
    "installment",
    "grade",
    "sub_grade",
    "emp_length",
    "home_ownership",
    "annual_inc",
    "verification_status",
    "issue_d",
    "loan_status",
    "purpose",
    "addr_state",
    "dti",
    "delinq_2yrs",
    "earliest_cr_line",
    "inq_last_6mths",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "total_pymnt",
    "total_rec_prncp",
    "recoveries",
    "out_prncp",
    "pub_rec_bankruptcies",
]

POLICY_PREFIX = "Does not meet the credit policy. Status:"


def _read_vintage(path: Path) -> pd.DataFrame:
    """Read one archive, keeping only the columns the study needs."""
    with zipfile.ZipFile(path) as archive:
        name = archive.namelist()[0]
        payload = archive.read(name).decode("latin-1")

    # The first line is a prospectus notice, not part of the table.
    first_line = payload.split("\n", 1)[0]
    skiprows = 0 if first_line.lstrip().startswith(('"id"', "id,")) else 1

    frame = pd.read_csv(
        io.StringIO(payload),
        skiprows=skiprows,
        usecols=lambda c: c in RAW_COLUMNS,
        low_memory=False,
    )
    # LendingClub appends summary lines at the foot of each file.
    return frame.dropna(subset=["loan_status"]).copy()


def _percent_to_float(series: pd.Series) -> pd.Series:
    """Turn ' 12.39%' into 12.39."""
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def _term_to_months(series: pd.Series) -> pd.Series:
    """Turn ' 36 months' into 36."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )


def _emp_length_to_years(series: pd.Series) -> pd.Series:
    """Turn '10+ years', '< 1 year' and 'n/a' into a number of years.

    The ceiling at ten and the floor at zero are the coding LendingClub itself
    uses; nothing finer is recoverable from the field.
    """
    text = series.astype(str).str.strip()
    years = pd.Series(np.nan, index=series.index, dtype=float)
    years = years.mask(text.str.startswith("10+"), 10.0)
    years = years.mask(text.str.startswith("< 1"), 0.0)
    plain = text.str.extract(r"^(\d+)\s+year", expand=False)
    years = years.fillna(pd.to_numeric(plain, errors="coerce"))
    return years


def _months_between(earlier: pd.Series, later: pd.Series) -> pd.Series:
    """Whole months separating two month-resolution dates."""
    return (later.dt.year - earlier.dt.year) * 12 + (later.dt.month - earlier.dt.month)


def _parse_month(series: pd.Series) -> pd.Series:
    """Parse the two month formats LendingClub mixes across vintages."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%b-%Y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(
            text[missing], format="%b-%y", errors="coerce"
        )
    return parsed


def build() -> pd.DataFrame:
    """Read every vintage and return one tidy modelling table."""
    frames = []
    for name, window in VINTAGES.items():
        path = RAW_DIR / f"{name}.csv.zip"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run src/download_data.py first")
        frame = _read_vintage(path)
        frame["vintage"] = window
        frame["source_file"] = name
        print(f"  {name} ({window}): {len(frame):,} loans".replace(",", " "))
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    # --- target -----------------------------------------------------------
    status = data["loan_status"].astype(str).str.strip()
    data["outside_credit_policy"] = status.str.startswith(POLICY_PREFIX).astype(int)
    status = status.str.replace(POLICY_PREFIX, "", regex=False).str.strip()
    data["loan_status_clean"] = status

    recognised = status.isin(["Fully Paid", "Charged Off"])
    dropped = int((~recognised).sum())
    if dropped:
        print(f"  dropping {dropped:,} loans with a non-terminal status".replace(",", " "))
    data = data.loc[recognised].copy()
    data["default"] = (data["loan_status_clean"] == "Charged Off").astype(int)

    # --- origination variables -------------------------------------------
    data["term_months"] = _term_to_months(data["term"])
    data["int_rate"] = _percent_to_float(data["int_rate"])
    data["revol_util"] = _percent_to_float(data["revol_util"])
    data["emp_length_years"] = _emp_length_to_years(data["emp_length"])

    issue_date = _parse_month(data["issue_d"])
    earliest = _parse_month(data["earliest_cr_line"])
    data["issue_date"] = issue_date
    data["issue_year"] = issue_date.dt.year
    data["credit_history_months"] = _months_between(earliest, issue_date)
    # A credit file cannot predate itself; treat impossible values as unknown.
    data.loc[data["credit_history_months"] < 0, "credit_history_months"] = np.nan

    for column in ("loan_amnt", "funded_amnt", "annual_inc", "dti", "installment"):
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # term is a contract length with two values; it reads better as a class.
    data["term_months"] = data["term_months"].astype("Int64").astype(str)

    # --- loss and exposure, observed only on defaults ---------------------
    funded = data["funded_amnt"].replace(0, np.nan)
    data["recovery_rate"] = (data["recoveries"] / funded).clip(lower=0, upper=1)
    # Credit conversion factor: the share of the granted amount still owed
    # when the loan defaults.
    data["ccf"] = ((funded - data["total_rec_prncp"]) / funded).clip(lower=0, upper=1)
    data["ead_observed"] = data["ccf"] * data["funded_amnt"]
    data["lgd_observed"] = 1.0 - data["recovery_rate"]
    for column in ("recovery_rate", "ccf", "ead_observed", "lgd_observed"):
        data.loc[data["default"] == 0, column] = np.nan

    keep = (
        ["default", "issue_date", "issue_year", "vintage", "source_file",
         "outside_credit_policy", "loan_status_clean"]
        + NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
        + LENDER_ASSESSMENT
        + ["funded_amnt", "recovery_rate", "ccf", "ead_observed", "lgd_observed"]
    )
    keep = [c for c in dict.fromkeys(keep) if c in data.columns]
    return data[keep].reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Reading raw archives:")
    data = build()

    destination = OUT_DIR / "loans.parquet"
    try:
        data.to_parquet(destination, index=False)
    except (ImportError, ValueError):
        destination = OUT_DIR / "loans.csv"
        data.to_csv(destination, index=False)

    defaults = int(data["default"].sum())
    print(f"\nModelling table: {len(data):,} loans, {data.shape[1]} columns".replace(",", " "))
    print(f"Defaults: {defaults:,} ({defaults / len(data):.2%})".replace(",", " "))
    print("\nDefault rate by vintage:")
    by_vintage = data.groupby("vintage")["default"].agg(["size", "mean"])
    for vintage, row in by_vintage.iterrows():
        print(f"  {vintage:<12} n = {int(row['size']):>7,}".replace(",", " ")
              + f"   default rate = {row['mean']:.2%}")
    print(f"\nWritten to {destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
