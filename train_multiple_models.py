import argparse
import time
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from mlflow_utils import promote_new_model_if_better, get_model_versions

#############################################
# Preprocessing Function
#############################################
def preprocess_df(df, target_col):
    """
    For numeric columns, convert to numeric and fill missing with the column's median.
    For object columns that are purely alphabetic, fill missing with "abc" (in lowercase) and label encode.
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
    if not pd.api.types.is_numeric_dtype(y):
        y = y.astype("category").cat.codes
    X = df.drop(columns=[target_col])
    return X, y

#############################################
# Metrics Function
#############################################
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def train_models(args):
    mlflow.set_experiment(args.experiment_name)
    df = pd.read_csv(args.csv_path)
    X, y = preprocess_df(df, args.target)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size/100.0, random_state=args.random_state
    )
    
    models = {}
    models["LogisticRegression"] = LogisticRegression(
        penalty=args.lr_penalty,
        C=args.lr_C,
        fit_intercept=args.lr_fit_intercept,
        random_state=args.random_state,
        max_iter=1000
    )
    models["SVC"] = SVC(
        kernel=args.svc_kernel,
        C=args.svc_C,
        probability=True,
        random_state=args.random_state
    )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=args.rf_n_estimators,
        max_depth=args.rf_max_depth,
        random_state=args.random_state
    )
    models["XGBoost"] = XGBClassifier(
        n_estimators=args.xgb_n_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=args.random_state
    )
    models["KNN"] = KNeighborsClassifier(n_neighbors=args.knn_n_neighbors)
    
    all_metrics = {}
    for model_name, model_obj in models.items():
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("random_state", args.random_state)
            if model_name == "LogisticRegression":
                mlflow.log_param("penalty", args.lr_penalty)
                mlflow.log_param("C", args.lr_C)
                mlflow.log_param("fit_intercept", args.lr_fit_intercept)
            elif model_name == "SVC":
                mlflow.log_param("kernel", args.svc_kernel)
                mlflow.log_param("C", args.svc_C)
            elif model_name == "RandomForest":
                mlflow.log_param("n_estimators", args.rf_n_estimators)
                mlflow.log_param("max_depth", args.rf_max_depth)
            elif model_name == "XGBoost":
                mlflow.log_param("n_estimators", args.xgb_n_estimators)
                mlflow.log_param("max_depth", args.xgb_max_depth)
                mlflow.log_param("learning_rate", args.xgb_learning_rate)
            elif model_name == "KNN":
                mlflow.log_param("n_neighbors", args.knn_n_neighbors)
            
            model_obj.fit(X_train, y_train)
            y_pred = model_obj.predict(X_test)
            metrics = compute_metrics(y_test, y_pred)
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            all_metrics[model_name] = metrics
            
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model_obj, artifact_path="model", registered_model_name=args.registered_model_name)
            else:
                mlflow.sklearn.log_model(model_obj, artifact_path="model", registered_model_name=args.registered_model_name)
            
            run_id = run.info.run_id
            mlflow.end_run()
            model_uri = f"runs:/{run_id}/model"
            reg_model_details = mlflow.register_model(model_uri=model_uri, name=args.registered_model_name)
            new_version = reg_model_details.version
            promote_new_model_if_better(args.registered_model_name, new_version, run_id, metric_key="f1")
    
    print("All models trained. Metrics summary:")
    for model_name, metrics in all_metrics.items():
        print(f"{model_name}: {metrics}")
    versions = get_model_versions(args.registered_model_name)
    print("Registered model versions for", args.registered_model_name)
    for v in versions:
        print(f"Version: {v.version}, Stage: {v.current_stage}, RunID: {v.run_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--test_size", type=float, default=20.0, help="Test size percentage")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--experiment_name", type=str, default="Default_Experiment", help="Experiment name to use")
    
    parser.add_argument("--lr_penalty", type=str, default="l2", help="Penalty for LogisticRegression")
    parser.add_argument("--lr_C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression")
    parser.add_argument("--lr_fit_intercept", type=lambda x: (str(x).lower() == 'true'), default=True, help="Whether to fit an intercept for LogisticRegression")
    
    parser.add_argument("--svc_kernel", type=str, default="rbf", help="Kernel for SVC")
    parser.add_argument("--svc_C", type=float, default=1.0, help="Regularization parameter C for SVC")
    
    parser.add_argument("--rf_n_estimators", type=int, default=100, help="Number of trees for RandomForestClassifier")
    parser.add_argument("--rf_max_depth", type=int, default=10, help="Max depth for RandomForestClassifier")
    
    parser.add_argument("--xgb_n_estimators", type=int, default=100, help="Number of trees for XGBoostClassifier")
    parser.add_argument("--xgb_max_depth", type=int, default=6, help="Max depth for XGBoostClassifier")
    parser.add_argument("--xgb_learning_rate", type=float, default=0.1, help="Learning rate for XGBoostClassifier")
    
    parser.add_argument("--knn_n_neighbors", type=int, default=5, help="Number of neighbors for KNeighborsClassifier")
    
    parser.add_argument("--registered_model_name", type=str, default="Classification_Champion", help="Registered model name in MLflow")
    
    args = parser.parse_args()
    train_models(args)
