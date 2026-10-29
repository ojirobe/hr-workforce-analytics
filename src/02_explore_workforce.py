"""Create reproducible workforce exploration outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"


def attrition_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    summary = (
        df.assign(attrition_flag=df["Attrition"].eq("Yes").astype(int))
        .groupby(column, observed=True)
        .agg(employees=("Attrition", "size"), attrition_count=("attrition_flag", "sum"), attrition_rate=("attrition_flag", "mean"))
        .reset_index()
    )
    summary["attrition_rate"] = (100 * summary["attrition_rate"]).round(2)
    return summary.sort_values("attrition_rate", ascending=False)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) == 1]
    categorical_columns = df.select_dtypes(include="object").columns.tolist()
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    quality = pd.DataFrame({
        "metric": ["rows", "columns", "missing_cells", "duplicate_rows", "constant_columns", "categorical_columns", "numeric_columns"],
        "value": [len(df), df.shape[1], int(df.isna().sum().sum()), int(df.duplicated().sum()), len(constant_columns), len(categorical_columns), len(numeric_columns)],
    })
    quality.to_csv(REPORTS / "data_quality_summary.csv", index=False)
    df[numeric_columns].describe().T.to_csv(REPORTS / "numeric_summary.csv")

    overtime = attrition_summary(df, "OverTime")
    department = attrition_summary(df, "Department")
    job_role = attrition_summary(df, "JobRole")
    overtime.to_csv(REPORTS / "attrition_by_overtime.csv", index=False)
    department.to_csv(REPORTS / "attrition_by_department.csv", index=False)
    job_role.to_csv(REPORTS / "attrition_by_job_role.csv", index=False)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(data=df, x="Attrition", hue="Attrition", legend=False, ax=axes[0], palette="Set2")
    axes[0].set_title("Workforce attrition counts")
    sns.barplot(data=overtime, x="OverTime", y="attrition_rate", hue="OverTime", legend=False, ax=axes[1], palette="Set2")
    axes[1].set_title("Attrition rate by overtime")
    axes[1].set_ylabel("Attrition rate (%)")
    fig.tight_layout()
    fig.savefig(FIGURES / "attrition_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Constant columns ({len(constant_columns)}): {constant_columns}")
    print(f"Numeric columns: {len(numeric_columns)} | Categorical columns: {len(categorical_columns)}")
    print("\nAttrition by overtime:")
    print(overtime.to_string(index=False))
    print("\nHighest-risk job roles:")
    print(job_role.head(3).to_string(index=False))
    print(f"\nSaved {5} tables and 1 figure under reports/.")


if __name__ == "__main__":
    main()
