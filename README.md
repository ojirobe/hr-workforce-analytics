# HR Workforce Analytics

An end-to-end analytics portfolio project examining employee attrition patterns with SQL, statistical testing, leakage-safe machine learning, automated tests, and an interactive Streamlit dashboard.

> **Important:** The dataset is synthetic. This project demonstrates analytical methods and must not be used to make employment decisions about individuals.

## Project overview

This project answers four practical questions:

1. What does the workforce look like across departments, roles, age, income, tenure, and overtime?
2. Which employee characteristics are statistically associated with attrition?
3. How accurately can attrition be estimated on unseen synthetic records without data leakage?
4. How can the findings be communicated through a responsible, interactive dashboard?

## Key results

- **1,470** synthetic employee records and **35** original fields
- **237** attrition cases: **16.12%** of the workforce
- No missing values and no duplicate rows
- Overtime employees show **30.5%** attrition versus **10.4%** for employees without overtime
- All eight preselected statistical tests remain significant after Holm correction
- Strongest categorical associations: overtime and job role, with Cramér's V near **0.24**
- Logistic regression selected by five-fold cross-validation: mean ROC AUC **0.827**
- Untouched holdout performance: ROC AUC **0.803**, average precision **0.561**, recall **0.638**

Predictive performance describes association in this synthetic dataset; it does not establish causality.

## Interactive dashboard

The Streamlit dashboard includes:

- Department, job role, overtime, and age filters
- Workforce and attrition KPI cards
- Department and job-role comparisons
- Age, income, and overtime visualisations
- Model-signal chart based on absolute logistic coefficients
- Filtered record table and CSV download
- Methodology and responsible-use guidance

Run it locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open `http://localhost:8501`.

## Reproduce the analysis

Run each numbered stage from the project root:

```bash
python src/01_validate_dataset.py
python src/02_explore_workforce.py
python src/03_run_sql_analysis.py
python src/04_statistical_tests.py
python src/05_train_models.py
pytest -q
```

The scripts generate processed data, CSV reports, figures, SQL outputs, model metrics, and the local serialized pipeline. Raw data and serialized models are intentionally excluded from Git.

## Methodology

### Data validation

The validation stage checks shape, schema, missingness, duplicates, target values, constant fields, numerical ranges, and class balance before analysis begins.

### SQL analysis

SQLite queries calculate overall attrition, department and role rates, overtime comparisons, income bands, age bands, tenure bands, satisfaction summaries, and travel patterns. Query outputs are exported to `reports/sql/`.

### Statistical testing

Categorical variables use chi-square tests with Cramér's V. Continuous variables use Welch's t-test with Hedges' g. Holm correction controls the family-wise error rate across the preselected hypotheses.

### Leakage-safe modelling

The model workflow:

- Splits an untouched 20% stratified holdout before model selection
- Removes identifiers, constants, and leakage-prone fields
- Fits preprocessing only inside scikit-learn pipelines
- Compares logistic regression and random forest using stratified five-fold cross-validation
- Selects the model using cross-validated ROC AUC
- Evaluates the selected model once on the untouched holdout

The dashboard presents aggregate signals only. It does not expose individual risk scores.

## Repository structure

```text
hr-workforce-analytics/
├── app.py                         # Streamlit Cloud entry point
├── dashboards/app.py              # Interactive dashboard
├── data/SOURCE.md                  # Dataset attribution and acquisition notes
├── reports/                        # Generated analytical outputs
├── sql/01_workforce_analysis.sql   # Reproducible SQL questions
├── src/                            # Numbered analysis pipeline
├── tests/test_project_contracts.py # Automated data and result contracts
├── requirements.txt               # Pinned Python dependencies
└── README.md
```

## Dataset and attribution

This project uses the **IBM HR Analytics Employee Attrition & Performance** synthetic dataset, commonly distributed as `WA_Fn-UseC_-HR-Employee-Attrition.csv`. The data contains no real IBM employee records. Acquisition details, provenance notes, and responsible-use limitations are documented in [data/SOURCE.md](data/SOURCE.md).

The raw CSV is not committed. Place it at:

```text
data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
```

## Automated tests

The seven project-contract tests verify:

- Raw dataset shape, completeness, uniqueness, and target counts
- Expected constant columns
- SQL reconciliation with the source data
- Department totals
- Statistical correction and effect-size bounds
- Model-selection and holdout metrics
- Leakage exclusions and pipeline structure

Run them with:

```bash
pytest -q
```

## Streamlit Community Cloud deployment

1. Fork or clone the repository.
2. Add the raw CSV at the documented path if it is not supplied securely during deployment.
3. In Streamlit Community Cloud, select this repository and branch.
4. Set the main file path to `app.py`.
5. Deploy with the pinned `requirements.txt`.

No secrets are required by the application.

## Responsible use

- This is a synthetic educational dataset, not evidence about IBM's real workforce.
- Attrition associations do not prove causes.
- Protected characteristics and proxy effects require careful fairness review.
- Model output must not drive individual hiring, promotion, discipline, or termination decisions.
- Appropriate use is aggregate exploration, method demonstration, and policy-question generation.

## Technology

Python, pandas, NumPy, SciPy, scikit-learn, SQLite, Matplotlib, Seaborn, Plotly, Streamlit, pytest, and Git.

## Author

**Yunusa Adavi Ojirobe**  
GitHub: [@ojirobe](https://github.com/ojirobe)
