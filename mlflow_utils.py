import mlflow
from mlflow.tracking import MlflowClient
from typing import List, Dict, Union, Optional

# Set MLflow to local server
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

def get_experiment_id_by_name(experiment_name: str) -> Optional[str]:
    """Return the experiment ID for a given experiment name, or None if not found."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment.experiment_id if experiment else None

def get_all_runs(experiment_name: str) -> List[Dict]:
    """Fetch all runs in a given experiment by name, sorted by start_time desc."""
    exp_id = get_experiment_id_by_name(experiment_name)
    if exp_id is None:
        return []
    
    runs = mlflow.search_runs(
        experiment_ids=[exp_id],
        order_by=["start_time DESC"]
    )
    return runs.to_dict(orient="records")  # Convert to list of dictionaries

def get_run_metrics(run_id: str) -> Dict[str, float]:
    """Return a dictionary of all available run metrics."""
    try:
        run_data = client.get_run(run_id).data
        return run_data.metrics  # Fetch all available metrics
    except Exception as e:
        print(f"Error fetching metrics for run {run_id}: {e}")
        return {}

def compare_runs_metrics(run_ids: list):
    """
    Fetch all available metrics for a list of run_ids and return them in a dict.
    """
    return {run_id: get_run_metrics(run_id) for run_id in run_ids}

def get_model_versions(model_name: str):
    """List all versions of a registered model."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        return versions  # Return ModelVersion objects (access via attributes)
    except Exception as e:
        print(f"Error fetching model versions for {model_name}: {e}")
        return []

def transition_model_version(model_name: str, version: int, stage: str):
    """Transition a specific version of a model to a new stage."""
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}.")
    except Exception as e:
        print(f"Error transitioning model {model_name} version {version} to {stage}: {e}")

def get_production_model_version(model_name: str):
    """Return the production model version (int) if it exists, else None."""
    versions = get_model_versions(model_name)
    for v in versions:
        if v.current_stage == "Production":
            return v.version
    return None

def get_metric_history(model_name: str, metric_key: str):
    """
    Get the history of a specific metric (e.g., 'f1') across all versions
    of the model by looking up each version's run_id.
    """
    versions = get_model_versions(model_name)
    history = []
    for v in versions:
        run_id = v.run_id
        run_metrics = get_run_metrics(run_id)
        if metric_key in run_metrics:
            history.append({
                "version": v.version,
                "stage": v.current_stage,
                "value": run_metrics[metric_key]
            })
    return history

def promote_new_model_if_better(model_name, new_version, new_run_id, metric_key="f1", higher_is_better=True):
    """
    Compare all registered versions for a given model and set the best-performing one to Production (Champion)
    while assigning all other versions to Staging (Challengers).
    """
    versions = get_model_versions(model_name)
    best_version = None
    best_metric = None
    for v in versions:
        run_id = v.run_id
        metrics = get_run_metrics(run_id)
        m = metrics.get(metric_key)
        if m is None:
            continue
        try:
            m = float(m)
        except:
            m = 0.0
        # If no best metric or we find a better one, update
        if best_metric is None or (higher_is_better and m > best_metric) or (not higher_is_better and m < best_metric):
            best_metric = m
            best_version = v.version
    # Transition the best to Production, others to Staging
    for v in versions:
        if v.version == best_version:
            if v.current_stage != "Production":
                transition_model_version(model_name, v.version, "Production")
                print(f"Version {v.version} set as Champion (Production).")
        else:
            if v.current_stage != "Staging":
                transition_model_version(model_name, v.version, "Staging")
                print(f"Version {v.version} set as Challenger (Staging).")

def list_experiments() -> Union[List[str], Dict[str, str]]:
    """Fetch all MLflow experiments as a list of experiment names."""
    try:
        experiments = client.search_experiments()
        return [exp.name for exp in experiments]
    except Exception as e:
        return {"error": str(e)}

def list_registered_models():
    """Fetch a live list of registered model names."""
    try:
        if hasattr(client, "list_registered_models"):
            models = client.list_registered_models()
            return [rm.name for rm in models]
        else:
            # Fallback: search all model versions and return unique model names.
            versions = client.search_model_versions("")
            names = set()
            for v in versions:
                names.add(v.name)
            return list(names)
    except Exception as e:
        return {"error": str(e)}
