import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

st.set_page_config(page_title="Career Progression Dashboard", layout="wide")
st.title("Career Progression & Promotion Gap Analysis")
st.caption("Exploratory employee analytics dashboard")
st.info("Verify the dataset source before attributing records to Palo Alto Networks. These analyses show associations, not causes.")

file = st.file_uploader("Upload employee CSV", type=["csv"])
if file is None:
    st.warning("Upload the CSV to begin. Required columns: Attrition, Department, YearsAtCompany, YearsSinceLastPromotion, YearsInCurrentRole, TrainingTimesLastYear.")
    st.stop()

try:
    df = pd.read_csv(file)
except Exception as e:
    st.error(f"Could not read CSV: {e}")
    st.stop()

required = ["Attrition", "Department", "YearsAtCompany", "YearsSinceLastPromotion", "YearsInCurrentRole", "TrainingTimesLastYear"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.write("Columns found:", list(df.columns))
    st.stop()

if pd.api.types.is_numeric_dtype(df["Attrition"]):
    df["AttritionBinary"] = pd.to_numeric(df["Attrition"], errors="coerce")
else:
    values = df["Attrition"].astype(str).str.strip().str.lower()
    df["AttritionBinary"] = values.map({"yes":1, "no":0, "left":1, "stayed":0, "true":1, "false":0, "1":1, "0":0})
if df["AttritionBinary"].isna().any() or not set(df["AttritionBinary"].dropna().unique()).issubset({0,1}):
    st.error("Attrition must use 0/1 or Yes/No (1/Yes = left, 0/No = stayed).")
    st.stop()
df["AttritionBinary"] = df["AttritionBinary"].astype(int)

n, left = len(df), int(df["AttritionBinary"].sum())
a,b,c,d = st.columns(4)
a.metric("Employees", f"{n:,}")
b.metric("Employees who left", f"{left:,}")
c.metric("Overall attrition", f"{100*left/n:.2f}%" if n else "N/A")
d.metric("Departments", str(df["Department"].nunique()))

overview, promo_tab, cluster_tab, quality_tab = st.tabs(["Overview", "Promotion Gap", "Career Clusters", "Data Quality"])
with overview:
    st.subheader("Attrition by department")
    dept = df.groupby("Department", dropna=False).agg(
        EmployeeCount=("AttritionBinary","size"),
        EmployeesLeft=("AttritionBinary","sum"),
        AttritionRate=("AttritionBinary","mean")).reset_index()
    dept["AttritionRate"] *= 100
    dept = dept.sort_values("AttritionRate", ascending=False)
    st.dataframe(dept.round(2), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.barh(dept["Department"].astype(str), dept["AttritionRate"])
    ax.set_xlabel("Attrition rate (%)")
    ax.set_title("Attrition Rate by Department")
    ax.invert_yaxis()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.subheader("Data preview")
    st.dataframe(df.drop(columns=["AttritionBinary"], errors="ignore").head(10), use_container_width=True, hide_index=True)

with promo_tab:
    st.subheader("Attrition by time since last promotion")
    gap = pd.to_numeric(df["YearsSinceLastPromotion"], errors="coerce")
    df["PromotionGapGroup"] = pd.cut(gap, bins=[-np.inf,1,3,5,np.inf], labels=["0–1 years","2–3 years","4–5 years","More than 5 years"])
    promo = df.groupby("PromotionGapGroup", observed=False).agg(
        EmployeeCount=("AttritionBinary","size"),
        EmployeesLeft=("AttritionBinary","sum"),
        AttritionRate=("AttritionBinary","mean")).reset_index()
    promo["AttritionRate"] *= 100
    st.dataframe(promo.round(2), use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(promo["PromotionGapGroup"].astype(str), promo["AttritionRate"])
    ax.set_title("Employee Attrition by Time Since Last Promotion")
    ax.set_ylabel("Attrition rate (%)")
    ax.set_xlabel("Time since last promotion")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("These groups are descriptive; they do not establish whether an employee is overdue for promotion.")

with cluster_tab:
    st.subheader("Exploratory career clusters")
    st.write("Uses three exploratory ratios. Training intensity is only a rough proxy and should not be treated as a validated metric.")
    tenure = pd.to_numeric(df["YearsAtCompany"], errors="coerce").replace(0, np.nan)
    feats = pd.DataFrame({
        "PromotionGapRatio": pd.to_numeric(df["YearsSinceLastPromotion"], errors="coerce") / tenure,
        "RoleStagnationIndex": pd.to_numeric(df["YearsInCurrentRole"], errors="coerce") / tenure,
        "TrainingIntensity": pd.to_numeric(df["TrainingTimesLastYear"], errors="coerce") / tenure,
    }).replace([np.inf,-np.inf], np.nan)
    k = st.selectbox("Number of clusters", [2,3,4], index=1)
    if len(df) <= k:
        st.warning("Not enough rows for this number of clusters.")
    else:
        X = SimpleImputer(strategy="median").fit_transform(feats)
        X = StandardScaler().fit_transform(X)
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        df["CareerCluster"] = labels
        st.metric("Silhouette score", f"{silhouette_score(X, labels):.3f}")
        profile = df.groupby("CareerCluster").agg(
            EmployeeCount=("AttritionBinary","size"),
            AvgYearsAtCompany=("YearsAtCompany","mean"),
            AvgYearsSincePromotion=("YearsSinceLastPromotion","mean"),
            AvgYearsInCurrentRole=("YearsInCurrentRole","mean"),
            AvgTrainingLastYear=("TrainingTimesLastYear","mean"),
            AttritionRate=("AttritionBinary","mean")).reset_index()
        profile["AttritionRate"] *= 100
        st.dataframe(profile.round(2), use_container_width=True, hide_index=True)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(profile["CareerCluster"].astype(str), profile["AttritionRate"])
        ax.set_title("Attrition Rate by Career Cluster")
        ax.set_ylabel("Attrition rate (%)")
        ax.set_xlabel("Career cluster")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

with quality_tab:
    st.subheader("Data quality")
    original = df.drop(columns=["AttritionBinary","PromotionGapGroup","CareerCluster"], errors="ignore")
    st.write({"Rows": len(original), "Columns": len(original.columns), "Duplicate rows": int(original.duplicated().sum())})
    st.dataframe(original.isna().sum().sort_values(ascending=False).rename("Missing values").to_frame(), use_container_width=True)

st.divider()
st.caption("Use these results to identify questions for further investigation; do not infer causation from descriptive rates.")
