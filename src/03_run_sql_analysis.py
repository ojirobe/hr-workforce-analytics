"""Build SQLite database, execute SQL views, and export verified results."""
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).parents[1]
CSV_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
DB_PATH = ROOT / "data/processed/hr_workforce.db"
SQL_PATH = ROOT / "sql/01_workforce_analysis.sql"
OUTPUT_DIR = ROOT / "reports/sql"
VIEWS = ["overall_attrition", "attrition_by_department", "attrition_by_overtime", "attrition_by_job_role", "income_by_attrition", "tenure_by_attrition"]


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    employees = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as connection:
        employees.to_sql("employees", connection, if_exists="replace", index=False)
        connection.executescript(SQL_PATH.read_text())
        exported = {}
        for view in VIEWS:
            result = pd.read_sql_query(f"SELECT * FROM {view}", connection)
            result.to_csv(OUTPUT_DIR / f"{view}.csv", index=False)
            exported[view] = result
        row_count = connection.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
        assert row_count == 1470
        assert len(exported) == 6
        assert float(exported["overall_attrition"].loc[0, "attrition_rate"]) == 16.12

    print(f"SQLite database rows: {row_count:,}")
    print(f"Executed and exported {len(exported)} SQL views.")
    print("\nOverall attrition:")
    print(exported["overall_attrition"].to_string(index=False))
    print("\nDepartment comparison:")
    print(exported["attrition_by_department"].to_string(index=False))
    print("\nIncome comparison (units; source currency unspecified):")
    print(exported["income_by_attrition"].to_string(index=False))


if __name__ == "__main__":
    main()
