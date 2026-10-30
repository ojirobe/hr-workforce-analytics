from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "employee_attrition.csv",
    PROJECT_ROOT / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
]
FEATURE_PATH = PROJECT_ROOT / "reports" / "model_feature_importance.csv"

st.set_page_config(
    page_title="HR Workforce Analytics",
    page_icon="📊",
    layout="wide",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    data_path = next(path for path in DATA_CANDIDATES if path.exists())
    data = pd.read_csv(data_path)
    data["AttritionFlag"] = data["Attrition"].eq("Yes").astype(int)
    return data


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    if not FEATURE_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURE_PATH)


def attrition_summary(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    return (
        frame.groupby(group, observed=True)
        .agg(Employees=("EmployeeNumber", "count"), Attrition=("AttritionFlag", "sum"))
        .assign(AttritionRate=lambda x: 100 * x["Attrition"] / x["Employees"])
        .reset_index()
        .sort_values("AttritionRate", ascending=False)
    )


data = load_data()

st.title("HR Workforce Analytics")
st.caption(
    "Explore workforce patterns in IBM's synthetic employee attrition dataset. "
    "Filters update every metric and chart on this page."
)

with st.sidebar:
    st.header("Workforce filters")
    departments = st.multiselect(
        "Department",
        sorted(data["Department"].unique()),
        default=sorted(data["Department"].unique()),
    )
    roles_available = sorted(data.loc[data["Department"].isin(departments), "JobRole"].unique())
    roles = st.multiselect("Job role", roles_available, default=roles_available)
    overtime = st.multiselect(
        "Overtime",
        sorted(data["OverTime"].unique()),
        default=sorted(data["OverTime"].unique()),
    )
    age_min, age_max = int(data["Age"].min()), int(data["Age"].max())
    age_range = st.slider("Age range", age_min, age_max, (age_min, age_max))
    st.divider()
    st.markdown("**Dataset:** 1,470 synthetic employee records")
    st.markdown("**Source:** IBM HR Analytics sample dataset")

filtered = data[
    data["Department"].isin(departments)
    & data["JobRole"].isin(roles)
    & data["OverTime"].isin(overtime)
    & data["Age"].between(*age_range)
].copy()

if filtered.empty:
    st.warning("No employees match these filters. Widen your selection to continue.")
    st.stop()

employees = len(filtered)
attrition_count = int(filtered["AttritionFlag"].sum())
attrition_rate = 100 * filtered["AttritionFlag"].mean()
avg_age = filtered["Age"].mean()
avg_tenure = filtered["YearsAtCompany"].mean()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Employees", f"{employees:,}")
k2.metric("Attrition cases", f"{attrition_count:,}")
k3.metric("Attrition rate", f"{attrition_rate:.1f}%")
k4.metric("Average age", f"{avg_age:.1f}")
k5.metric("Average tenure", f"{avg_tenure:.1f} yrs")

st.divider()
overview_tab, drivers_tab, data_tab = st.tabs(
    ["Workforce overview", "Attrition patterns", "Data & methodology"]
)

with overview_tab:
    left, right = st.columns(2)
    dept = attrition_summary(filtered, "Department")
    role = attrition_summary(filtered, "JobRole")

    with left:
        st.subheader("Attrition rate by department")
        fig = px.bar(
            dept,
            x="Department",
            y="AttritionRate",
            text="AttritionRate",
            color="AttritionRate",
            color_continuous_scale="Reds",
            labels={"AttritionRate": "Attrition rate (%)"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, max(25, dept["AttritionRate"].max() * 1.25)])
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Employees by job role")
        fig = px.bar(
            role.sort_values("Employees"),
            x="Employees",
            y="JobRole",
            orientation="h",
            color="AttritionRate",
            color_continuous_scale="Reds",
            labels={"AttritionRate": "Attrition rate (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Age and attrition")
    fig = px.histogram(
        filtered,
        x="Age",
        color="Attrition",
        barmode="overlay",
        nbins=25,
        color_discrete_map={"No": "#4C78A8", "Yes": "#E45756"},
        opacity=0.75,
    )
    st.plotly_chart(fig, use_container_width=True)

with drivers_tab:
    left, right = st.columns(2)
    with left:
        st.subheader("Overtime comparison")
        overtime_summary = attrition_summary(filtered, "OverTime")
        fig = px.bar(
            overtime_summary,
            x="OverTime",
            y="AttritionRate",
            text="AttritionRate",
            color="OverTime",
            labels={"OverTime": "Works overtime", "AttritionRate": "Attrition rate (%)"},
            color_discrete_map={"No": "#4C78A8", "Yes": "#E45756"},
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Monthly income distribution")
        fig = px.box(
            filtered,
            x="Attrition",
            y="MonthlyIncome",
            color="Attrition",
            color_discrete_map={"No": "#4C78A8", "Yes": "#E45756"},
            labels={"MonthlyIncome": "Monthly income"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Model signals—not causal effects")
    importance = load_feature_importance()
    if not importance.empty:
        value_col = "absolute_importance" if "absolute_importance" in importance.columns else importance.columns[-1]
        feature_col = "feature" if "feature" in importance.columns else importance.columns[0]
        top = importance.nlargest(12, value_col).sort_values(value_col)
        fig = px.bar(top, x=value_col, y=feature_col, orientation="h", labels={value_col: "Absolute coefficient", feature_col: "Feature"})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the modelling script to generate the feature-signal report.")
    st.info(
        "These associations support workforce planning and further investigation. "
        "They do not prove causation and must not be used to make decisions about individual employees."
    )

with data_tab:
    st.subheader("Filtered records")
    shown = ["EmployeeNumber", "Department", "JobRole", "Age", "OverTime", "MonthlyIncome", "YearsAtCompany", "Attrition"]
    st.dataframe(filtered[shown], use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        filtered[shown].to_csv(index=False),
        file_name="filtered_workforce.csv",
        mime="text/csv",
    )
    st.subheader("Responsible interpretation")
    st.markdown(
        """
- The dataset is synthetic and is not evidence about IBM's real workforce.
- Attrition is an observed association target, not a judgement of employee value.
- The predictive model uses a held-out test set and excludes obvious identifiers and constants.
- Results should guide questions and aggregate policy review—not individual employment action.
        """
    )
