# train_random_forest.py
import mlflow
import mlflow.sklearn
import mlflow.system_metrics
import mlflow.shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import time

# For model registry
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change if using a remote MLflow server
mlflow.set_experiment("California_Housing_Regression")  # Custom experiment name

# Enable system metrics logging globally
mlflow.system_metrics.enable_system_metrics_logging()
mlflow.system_metrics.set_system_metrics_samples_before_logging(5)
mlflow.system_metrics.set_system_metrics_sampling_interval(2)

def train_and_log_random_forest():
    # Load Online Dataset (California Housing)
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="Price")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define model hyperparameters
    hyperparams = {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }

    # Start MLflow Run with System Metrics
    with mlflow.start_run(run_name="RandomForest_California_Housing", log_system_metrics=True) as run:

        # Log Dataset Details
        mlflow.log_param("dataset", "California Housing (from sklearn.datasets)")
        mlflow.log_param("features", list(X.columns))

        # Log Hyperparameters
        for param, value in hyperparams.items():
            mlflow.log_param(param, value)

        # Train Model
        model = RandomForestRegressor(**hyperparams)
        model.fit(X_train, y_train)

        # Make Predictions
        y_pred = model.predict(X_test)

        # Compute Metrics
        metrics = {
            "MSE": mean_squared_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "MAE": mean_absolute_error(y_test, y_pred),
            "R2": r2_score(y_test, y_pred),
        }

        # Log Metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)

        # Log Model
        model_path = "random_forest_model"
        mlflow.sklearn.log_model(
            model, 
            model_path, 
            registered_model_name="RandomForest_California"
        )

        # Create an "artifacts" directory to store plots
        os.makedirs("artifacts", exist_ok=True)

        # Plot 1: Feature Importance
        plt.figure(figsize=(10, 5))
        feature_importance = pd.Series(
            model.feature_importances_, index=X.columns
        ).sort_values(ascending=False)
        sns.barplot(x=feature_importance.values, y=feature_importance.index)
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Features")
        feature_imp_path = "artifacts/feature_importance.png"
        plt.savefig(feature_imp_path)
        mlflow.log_artifact(feature_imp_path)

        # Plot 2: Predictions vs Actual
        plt.figure(figsize=(8, 5))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.title("Actual vs Predicted Prices")
        plt.grid(True)
        pred_vs_actual_path = "artifacts/pred_vs_actual.png"
        plt.savefig(pred_vs_actual_path)
        mlflow.log_artifact(pred_vs_actual_path)

        # Plot 3: Correlation Heatmap
        plt.figure(figsize=(10, 8))
        corr = X_train.corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Feature Correlation Heatmap")
        corr_path = "artifacts/correlation_heatmap.png"
        plt.savefig(corr_path)
        mlflow.log_artifact(corr_path)

        # Log Model Schema
        mlflow.log_text(
            "Dataset: California Housing (from sklearn)\nSource: sklearn.datasets\nTask: Regression",
            "artifacts/dataset_info.txt"
        )

        # Log Tags
        mlflow.set_tag("Project", "MLflow Experiment")
        mlflow.set_tag("Framework", "Scikit-Learn")
        mlflow.set_tag("Model_Type", "RandomForestRegressor")
        mlflow.set_tag("Source", "MLflow Experiment Script")

        # SHAP Explainer
        explainer = shap.Explainer(model.predict, X_train)
        shap_values = explainer(X_test)

        # Save and Log SHAP Summary Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, show=False)
        shap_plot_path = "artifacts/shap_summary.png"
        plt.savefig(shap_plot_path)
        mlflow.log_artifact(shap_plot_path)

        # Log SHAP Explainer
        mlflow.shap.log_explainer(
            explainer, 
            artifact_path="shap_explainer", 
            registered_model_name="RandomForest_SHAP_Explainer"
        )

        # Print run ID
        print(f"Run ID: {run.info.run_id}")

        # -------------- Model Registry: versioning & staging --------------
        # Register the newly logged model version under the same name used in log_model
        client = MlflowClient()
        model_uri = f"runs:/{run.info.run_id}/{model_path}"
        registered_name = "RandomForest_California"

        # Register model
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=registered_name
        )

        # Wait for the model to be ready
        time.sleep(5)

        # Transition model to 'Staging' (as an example)
        client.transition_model_version_stage(
            name=registered_name,
            version=model_details.version,
            stage="Staging"
        )
        print(f"Model version {model_details.version} transitioned to Staging.")

if __name__ == "__main__":
    train_and_log_random_forest()
