"""Test selected attrition associations with effect sizes and Holm correction."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
OUTPUT_PATH = ROOT / "reports/statistical_tests.csv"
CATEGORICAL = ["OverTime", "Department", "JobRole"]
NUMERIC = ["MonthlyIncome", "Age", "YearsAtCompany", "TotalWorkingYears", "DistanceFromHome"]


def cramers_v(table: pd.DataFrame, chi2: float) -> float:
    n = table.to_numpy().sum()
    denominator = n * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(chi2 / denominator)) if denominator else 0.0


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    results = []

    for feature in CATEGORICAL:
        table = pd.crosstab(df[feature], df["Attrition"])
        chi2, p_value, degrees_freedom, _ = chi2_contingency(table)
        results.append({
            "feature": feature, "test": "chi-square", "statistic": chi2,
            "degrees_freedom": degrees_freedom, "p_value": p_value,
            "effect_size": cramers_v(table, chi2), "effect_name": "Cramer's V",
            "attrition_group_summary": "category rates in SQL/EDA outputs",
        })

    for feature in NUMERIC:
        left = df.loc[df["Attrition"] == "Yes", feature]
        stayed = df.loc[df["Attrition"] == "No", feature]
        statistic, p_value = mannwhitneyu(left, stayed, alternative="two-sided")
        rank_biserial = 2 * statistic / (len(left) * len(stayed)) - 1
        results.append({
            "feature": feature, "test": "Mann-Whitney U", "statistic": statistic,
            "degrees_freedom": np.nan, "p_value": p_value,
            "effect_size": rank_biserial, "effect_name": "rank-biserial correlation",
            "attrition_group_summary": f"median left={left.median():.2f}; stayed={stayed.median():.2f}",
        })

    results_df = pd.DataFrame(results)
    reject, adjusted, _, _ = multipletests(results_df["p_value"], alpha=0.05, method="holm")
    results_df["p_value_holm"] = adjusted
    results_df["significant_05"] = reject
    results_df["effect_size"] = results_df["effect_size"].round(4)
    results_df = results_df.sort_values("p_value_holm")
    results_df.to_csv(OUTPUT_PATH, index=False)

    print("Holm-corrected statistical tests (association, not causation):")
    print(results_df[["feature", "test", "p_value_holm", "effect_name", "effect_size", "significant_05"]].to_string(index=False))
    print(f"\nSignificant after correction: {int(reject.sum())} of {len(results_df)} tests")
    print(f"Saved: {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
