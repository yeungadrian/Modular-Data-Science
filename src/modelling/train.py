import logging

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import OmegaConf
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from ray import tune
from ray.air import session
from ray.air.integrations.mlflow import setup_mlflow
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


conf = OmegaConf.load("config/main.yaml")


def create_experiment(experiment_name):
    try:
        mlflow.create_experiment(experiment_name)
    except Exception:
        logging.info("Experiment already exists")


@task
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


@task
def split_data(
    df: pd.DataFrame, features: list[str], target: str, test_size: float
) -> list[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target].values, test_size=test_size
    )
    return x_train, x_test, y_train, y_test


def train_model(config, x_train, y_train, eval_set):
    model = xgb.XGBClassifier(
        objective=config["objective"],
        max_depth=config["max_depth"],
        min_child_weight=config["min_child_weight"],
        subsample=config["subsample"],
        learning_rate=config["learning_rate"],
        num_class=3,
    )
    model.fit(x_train, y_train, eval_set=eval_set, verbose=False)
    return model


def validate_model(model, x_test, y_test):
    y_pred = model.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_pred, multi_class="ovo")
    metrics = {"auc": auc}
    return metrics


def run_experiment(config: dict, data):
    setup_mlflow(config)
    x_train, x_test, y_train, y_test = data
    model = train_model(
        config, x_train, y_train, [(x_train, y_train), (x_test, y_test)]
    )
    for key, value in model.evals_result().items():
        for n, i in enumerate(value["mlogloss"]):
            mlflow.log_metric(f"{key}-mlogloss", float(i), n)
    metrics = validate_model(model, x_test, y_test)
    mlflow.log_metrics(metrics)
    metrics["done"] = True
    session.report(metrics)


@task
def tune_model(data):
    search_space = {
        "objective": "multi:softmax",
        "max_depth": tune.randint(1, 9),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1.0),
        "learning_rate": tune.loguniform(1e-4, 1e-1),
        "mlflow": {
            "experiment_name": "iris-classifier",
            "tracking_uri": mlflow.get_tracking_uri(),
            "registry_uri": None,
        },
    }
    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)
    tuner = tune.Tuner(
        tune.with_parameters(run_experiment, data=data),
        tune_config=tune.TuneConfig(
            metric="auc",
            mode="max",
            scheduler=scheduler,
            num_samples=2,
        ),
        param_space=search_space,
    )
    results = tuner.fit()
    return results


@task
def write_search_results(results: tune.ResultGrid, path: str) -> None:
    results.get_dataframe().to_csv(path)
    return None


@flow(task_runner=SequentialTaskRunner())
def main() -> None:
    create_experiment(conf["mlflow"]["experiment_name"])
    df = load_data(conf["data"]["features"])
    x_train, x_test, y_train, y_test = split_data(
        df,
        conf["model"]["features"],
        conf["model"]["target"],
        conf["model"]["test_size"],
    )
    data = x_train, x_test, y_train, y_test
    results = tune_model(data)
    write_search_results(results, conf["data"]["hp_search"])
    return None


if __name__ == "__main__":
    main()
