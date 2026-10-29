"""Train and evaluate leakage-safe attrition classifiers."""
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parents[1]
DATA_PATH = ROOT / "data/raw/WA_Fn-UseC_-HR-Employee-Attrition.csv"
DROP_COLUMNS = ["Attrition", "EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
RANDOM_STATE = 42


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=DROP_COLUMNS)
    y = df["Attrition"].eq("Yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows, pipelines = [], {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessing", preprocessing), ("model", estimator)])
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc_mean": scores.mean(), "cv_roc_auc_std": scores.std()})
        pipelines[name] = pipeline

    metrics = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    selected_name = metrics.iloc[0]["model"]
    selected = pipelines[selected_name].fit(X_train, y_train)
    probabilities = selected.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    holdout = {
        "holdout_roc_auc": roc_auc_score(y_test, probabilities),
        "holdout_average_precision": average_precision_score(y_test, probabilities),
        "holdout_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
        "holdout_precision": precision_score(y_test, predictions),
        "holdout_recall": recall_score(y_test, predictions),
        "holdout_f1": f1_score(y_test, predictions),
    }
    metrics["selected"] = metrics["model"].eq(selected_name)
    for key, value in holdout.items():
        metrics[key] = value if metrics["selected"].any() else float("nan")
        metrics.loc[~metrics["selected"], key] = float("nan")

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports/figures").mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, ROOT / "models/attrition_pipeline.joblib")
    metrics.to_csv(ROOT / "reports/model_metrics.csv", index=False)

    feature_names = selected.named_steps["preprocessing"].get_feature_names_out()
    model = selected.named_steps["model"]
    values = model.coef_[0] if hasattr(model, "coef_") else model.feature_importances_
    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    importance["absolute_importance"] = importance["importance"].abs()
    importance.sort_values("absolute_importance", ascending=False).head(25).to_csv(ROOT / "reports/model_feature_importance.csv", index=False)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, probabilities)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(fpr, tpr, label=f"AUC={holdout['holdout_roc_auc']:.3f}")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="grey")
    axes[0].set(title="Holdout ROC curve", xlabel="False positive rate", ylabel="True positive rate")
    axes[0].legend()
    axes[1].plot(recall_curve, precision_curve, label=f"AP={holdout['holdout_average_precision']:.3f}")
    axes[1].set(title="Holdout precision-recall curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures/model_performance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Train rows: {len(X_train):,} | Untouched test rows: {len(X_test):,}")
    print("Cross-validation comparison (training partition only):")
    print(metrics[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "selected"]].to_string(index=False))
    print(f"\nSelected model: {selected_name}")
    for key, value in holdout.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    main()
