"""Validate the raw IBM HR attrition sample before analysis."""
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parents[1] / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
EXPECTED_COLUMNS = 35
EXPECTED_ROWS = 1470


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS), f"Unexpected shape: {df.shape}"
    assert df.duplicated().sum() == 0, "Duplicate rows detected"
    assert df.isna().sum().sum() == 0, "Missing values detected"
    assert set(df["Attrition"].unique()) == {"Yes", "No"}, "Unexpected target values"

    attrition_count = int((df["Attrition"] == "Yes").sum())
    attrition_rate = 100 * attrition_count / len(df)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]}")
    print(f"Missing cells: {int(df.isna().sum().sum())}")
    print(f"Duplicate rows: {int(df.duplicated().sum())}")
    print(f"Attrition: {attrition_count:,} employees ({attrition_rate:.2f}%)")


if __name__ == "__main__":
    main()
