import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pyfunc

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

import plotly.express as px

# Import from mlflow_utils
from mlflow_utils import (
    get_all_runs,
    list_experiments,
    list_registered_models,
    promote_new_model_if_better,
    get_model_versions
)

# Set local MLflow server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

st.title("Unified MLflow App (Regression & Classification)")

#############################################
# Preprocessing
#############################################
def preprocess_df(df, target_col, task_type):
    """
    For numeric columns, convert to numeric and fill missing with median.
    For object columns that are purely alphabetic, fill missing with "abc" in lowercase and label encode.
    Otherwise, attempt numeric conversion or label encode.
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype.kind in 'biufc':
            df[col] = pd.to_numeric(df[col], errors='coerce')
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = df[col].fillna("abc")
            if df[col].str.isalpha().all():
                df[col] = df[col].str.lower()
                df[col] = df[col].astype("category").cat.codes
            else:
                try:
                    df[col] = pd.to_numeric(df[col], errors='raise')
                except:
                    df[col] = df[col].astype("category").cat.codes
    y = df[target_col]
    if task_type == "Classification" and not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes
    X = df.drop(columns=[target_col])
    return X, y

#############################################
# Metric Computations
#############################################
def compute_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}

#############################################
# Visualization Helpers
#############################################
def plot_metrics_comparison(metrics_dict):
    if not metrics_dict:
        st.write("No metrics to compare.")
        return
    df = pd.DataFrame(metrics_dict).T
    fig, axes = plt.subplots(1, len(df.columns), figsize=(4 * len(df.columns), 4))
    if len(df.columns) == 1:
        axes = [axes]
    for ax, metric in zip(axes, df.columns):
        df[metric].plot(kind='bar', ax=ax, title=metric)
        ax.set_xlabel("Run ID")
        ax.set_ylabel(metric)
    plt.tight_layout()
    st.pyplot(fig)

def plot_parallel_coordinates(runs, param_names, metric_name="MAE"):
    if not runs:
        st.warning("No runs to plot.")
        return
    rows = []
    for r in runs:
        run_id = r["run_id"]
        row = {"run_id": run_id}
        # Collect parameters
        for p in param_names:
            row[p] = r.get(f"params.{p}", None)
        # Collect metric
        row[metric_name] = r.get(f"metrics.{metric_name}", None)
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in param_names:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category").cat.codes
    df[metric_name] = pd.to_numeric(df[metric_name], errors="coerce")
    df.dropna(subset=[metric_name], inplace=True)
    if df.empty:
        st.warning("No valid rows to plot after removing NaN metrics.")
        return
    fig = px.parallel_coordinates(
        df,
        color=metric_name,
        dimensions=param_names,
        color_continuous_scale=px.colors.sequential.Viridis,
        labels={metric_name: metric_name},
        title=f"Parallel Coordinates (colored by {metric_name})"
    )
    st.plotly_chart(fig)

#############################################
# Experiment & Model Setup
#############################################
st.header("Experiment Setup")
exp_option = st.radio("Experiment Option", ("Use Existing", "Create New"))
if exp_option == "Use Existing":
    existing_exps = list_experiments()
    if isinstance(existing_exps, dict) and "error" in existing_exps:
        st.error("Error fetching experiments: " + existing_exps["error"])
        exp_name = st.text_input("Enter Experiment Name", "Default_Experiment")
    else:
        exp_name = st.selectbox("Select Experiment", existing_exps)
else:
    exp_name = st.text_input("Enter New Experiment Name", value="New_Experiment")

if exp_name:
    mlflow.set_experiment(exp_name)

st.header("Model Registration")
reg_option = st.radio("Register Model Option", ("Use Existing", "Create New"))
if reg_option == "Use Existing":
    existing_models = list_registered_models()
    if isinstance(existing_models, dict) and "error" in existing_models:
        st.error("Error fetching registered models: " + existing_models["error"])
        reg_model_name = st.text_input("Enter Registered Model Name", value="ML_Champion")
    else:
        reg_model_name = st.selectbox("Select Registered Model", existing_models)
else:
    reg_model_name = st.text_input("Enter New Registered Model Name", value="ML_Champion")

#############################################
# Data Upload & Model Training
#############################################
st.header("Upload & Train")
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview:")
    st.dataframe(raw_df.head())

    task_type = st.selectbox("Select Task Type", ["Regression", "Classification"])
    target_col = st.selectbox("Select Target Column", raw_df.columns)
    df = raw_df.copy()
    X, y = preprocess_df(df, target_col, task_type)

    st.write("Preprocessed Data Preview:")
    st.dataframe(X.head())

    random_state = st.slider("Random State", 0, 999, 42, 1)
    test_size_percent = st.slider("Test Size %", 5, 50, 20, 5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size_percent/100.0, random_state=random_state
    )

    # Model Selection
    st.header("Choose Model & Hyperparams")
    if task_type == "Regression":
        model_options = ["RandomForestRegressor", "XGBRegressor"]
    else:
        model_options = ["LogisticRegression", "RandomForestClassifier", "XGBClassifier"]
    chosen_model = st.selectbox("Model", model_options)

    params = {}
    model = None

    # Regression
    if chosen_model == "RandomForestRegressor":
        n_estimators = st.slider("n_estimators", 10, 300, 100, 10)
        max_depth = st.slider("max_depth", 1, 50, 6, 1)
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**params)
    elif chosen_model == "XGBRegressor":
        n_estimators = st.slider("n_estimators", 10, 300, 100, 10)
        max_depth = st.slider("max_depth", 1, 50, 6, 1)
        learning_rate = st.slider("learning_rate", 0.001, 1.0, 0.1, 0.001)
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate, "random_state": random_state}
        from xgboost import XGBRegressor
        model = XGBRegressor(**params)

    # Classification
    if chosen_model == "LogisticRegression":
        penalty = st.selectbox("Penalty", ["l2", "none"])
        C = st.slider("Inverse Reg. Strength (C)", 0.01, 10.0, 1.0, 0.01)
        fit_intercept = st.checkbox("Fit Intercept?", value=True)
        params = {"penalty": penalty if penalty != "none" else "none", "C": C, "fit_intercept": fit_intercept, "random_state": random_state, "max_iter": 1000}
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(**params)
    elif chosen_model == "RandomForestClassifier":
        n_estimators = st.slider("n_estimators", 10, 300, 100, 10)
        max_depth = st.slider("max_depth", 1, 50, 6, 1)
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": random_state}
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**params)
    elif chosen_model == "XGBClassifier":
        n_estimators = st.slider("n_estimators", 10, 300, 100, 10)
        max_depth = st.slider("max_depth", 1, 50, 6, 1)
        learning_rate = st.slider("learning_rate", 0.001, 1.0, 0.1, 0.001)
        params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate, "use_label_encoder": False, "eval_metric": "logloss", "random_state": random_state}
        from xgboost import XGBClassifier
        model = XGBClassifier(**params)

    if st.button("Train & Register"):
        if model is None:
            st.error("No model selected.")
        else:
            with mlflow.start_run(run_name=f"{chosen_model}_run") as run:
                mlflow.log_param("task_type", task_type)
                mlflow.log_param("model_type", chosen_model)
                mlflow.log_param("random_state", random_state)
                mlflow.log_param("test_size_percent", test_size_percent)
                for k, v in params.items():
                    mlflow.log_param(k, v)

                # Train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Metrics
                if task_type == "Regression":
                    metrics = compute_regression_metrics(y_test, y_pred)
                else:
                    metrics = compute_classification_metrics(y_test, y_pred)
                for m_name, m_val in metrics.items():
                    mlflow.log_metric(m_name, m_val)

                # Log model
                if "XGB" in chosen_model:
                    mlflow.xgboost.log_model(model, "model", registered_model_name=reg_model_name)
                else:
                    mlflow.sklearn.log_model(model, "model", registered_model_name=reg_model_name)

                st.success(f"Trained {chosen_model} and logged to MLflow (run_id={run.info.run_id}).")
                st.write("Metrics:", metrics)

                # Register
                model_uri = f"runs:/{run.info.run_id}/model"
                reg_model_details = mlflow.register_model(model_uri=model_uri, name=reg_model_name)
                new_version = reg_model_details.version
                st.write(f"Registered new version: {new_version}")

                # Possibly wait for registration to complete
                time.sleep(3)

                # Promote if better
                if task_type == "Regression":
                    # We'll use "MAE" as the key, lower is better
                    promote_new_model_if_better(reg_model_name, new_version, run.info.run_id, metric_key="MAE", higher_is_better=False)
                else:
                    # We'll use "f1" as the key, higher is better
                    promote_new_model_if_better(reg_model_name, new_version, run.info.run_id, metric_key="f1", higher_is_better=True)

#############################################
# Compare Runs
#############################################
st.write("---")
st.header("Compare Runs in This Experiment")
if st.button("Fetch & Compare All Runs"):
    if exp_name:
        runs = get_all_runs(exp_name)
        if len(runs) == 0:
            st.write("No runs found in this experiment yet.")
        else:
            st.write(f"Found {len(runs)} runs in experiment '{exp_name}'.")
            run_metrics_dict = {}
            for r in runs:
                run_id = r["run_id"]
                # Extract "metrics." keys
                these_metrics = {key.replace("metrics.", ""): r[key] for key in r if key.startswith("metrics.")}
                run_metrics_dict[run_id] = these_metrics
            st.json(run_metrics_dict)
            plot_metrics_comparison(run_metrics_dict)
    else:
        st.info("Please set an experiment first.")

#############################################
# Parallel Coordinates
#############################################
st.write("---")
st.header("Parallel Coordinates Plot")
st.write("Enter comma-separated param names (e.g., max_depth,n_estimators) plus a metric (e.g. MAE, f1).")
param_input = st.text_input("Param names", value="max_depth,n_estimators")
color_metric = st.text_input("Metric name", value="MAE")
if st.button("Plot Parallel Coordinates"):
    if exp_name:
        runs = get_all_runs(exp_name)
        param_list = [p.strip() for p in param_input.split(",") if p.strip()]
        if len(runs) == 0:
            st.write("No runs to plot.")
        elif not param_list:
            st.write("Please provide at least one param name.")
        else:
            plot_parallel_coordinates(runs, param_list, metric_name=color_metric)
    else:
        st.info("Please set an experiment name first.")

#############################################
# Model Registry: Versions
#############################################
st.write("---")
st.header("Model Registry: Versions")
if st.button("List Model Versions"):
    versions = get_model_versions(reg_model_name)
    if versions:
        for v in versions:
            st.write(f"Version: {v.version}, Stage: {v.current_stage}, RunID: {v.run_id}")
    else:
        st.write(f"No versions found for '{reg_model_name}'.")
