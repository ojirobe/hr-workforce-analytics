# Dataset source and usage notes

## Dataset

**IBM HR Analytics Employee Attrition & Performance**

- Canonical listing: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
- Local filename: `raw/WA_Fn-UseC_-HR-Employee-Attrition.csv`
- Acquisition mirror used for this reproducible local setup: https://github.com/shantanu1109/IBM-HR-Analytics-Employee-Attrition-and-Performance-Prediction/blob/main/DATASET/IBM-HR-Analytics-Employee-Attrition-and-Performance.csv
- Retrieved: 2026-10-29

## Attribution and limitations

This is a fictional/synthetic sample dataset created by IBM data scientists for analytics practice. It must not be represented as IBM's real workforce data. The source contains no event dates, so this project will not invent time-series trends, hire dates, exit dates, or forecasts. Rate and income fields have no documented currency; dashboards will show them as numeric units without a currency symbol.

The dataset's exact redistribution license is not clearly stated on the canonical listing. Before publishing the repository, the raw CSV should be excluded unless its redistribution terms are confirmed; project code will support local acquisition.

## Executed validation

- File size: 227,977 bytes
- Shape: 1,470 rows x 35 columns
- Duplicate rows: 0
- Missing cells: 0
- Attrition: 237 Yes, 1,233 No
- Attrition rate: 16.12%
