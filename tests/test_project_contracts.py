"""Regression tests for data, SQL, statistics, and modelling contracts."""
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"


def test_raw_dataset_contract() -> None:
    df = pd.read_csv(DATA_PATH)
    assert df.shape == (1470, 35)
    assert df.isna().sum().sum() == 0
    assert df.duplicated().sum() == 0
    assert df["Attrition"].value_counts().to_dict() == {"No": 1233, "Yes": 237}


def test_constant_columns_are_known() -> None:
    df = pd.read_csv(DATA_PATH)
    constants = {column for column in df.columns if df[column].nunique(dropna=False) == 1}
    assert constants == {"EmployeeCount", "Over18", "StandardHours"}


def test_sql_exports_reconcile_with_source() -> None:
    overall = pd.read_csv(ROOT / "reports/sql/overall_attrition.csv")
    assert overall.loc[0, "employees"] == 1470
    assert overall.loc[0, "attrition_count"] == 237
    assert overall.loc[0, "attrition_rate"] == pytest.approx(16.12)


def test_sql_department_counts_cover_all_employees() -> None:
    departments = pd.read_csv(ROOT / "reports/sql/attrition_by_department.csv")
    assert departments["employees"].sum() == 1470
    assert departments["attrition_count"].sum() == 237


def test_statistical_results_are_valid_and_corrected() -> None:
    tests = pd.read_csv(ROOT / "reports/statistical_tests.csv")
    assert len(tests) == 8
    assert tests["p_value_holm"].between(0, 1).all()
    assert (tests["p_value_holm"] >= tests["p_value"] - 1e-15).all()
    assert tests["effect_size"].abs().le(1).all()


def test_model_selection_and_holdout_contract() -> None:
    metrics = pd.read_csv(ROOT / "reports/model_metrics.csv")
    assert len(metrics) == 2
    assert metrics["selected"].sum() == 1
    selected = metrics.loc[metrics["selected"]].iloc[0]
    rejected = metrics.loc[~metrics["selected"]].iloc[0]
    assert selected["model"] == "logistic_regression"
    assert selected["holdout_roc_auc"] == pytest.approx(0.803, abs=0.002)
    assert pd.isna(rejected["holdout_roc_auc"])


def test_leakage_prone_fields_are_explicitly_excluded() -> None:
    source = (ROOT / "src/05_train_models.py").read_text()
    for field in ["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]:
        assert field in source
    assert "train_test_split" in source
    assert "Pipeline" in source
    assert "cross_val_score" in source
