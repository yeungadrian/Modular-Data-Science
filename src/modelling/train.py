import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import OmegaConf
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

conf = OmegaConf.load("config/main.yaml")


@task
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@task
def split_data(
    df: pd.DataFrame, features: list[str], target: str, test_size: float
) -> list[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target].values, test_size=test_size
    )
    return x_train, x_test, y_train, y_test


@task
def train_best_model(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_path: str,
) -> None:
    mlflow.start_run()
    model = xgb.XGBClassifier()
    model.fit(x_train, y_train, eval_set=[(x_test, y_test)])
    params = model.get_params()
    mlflow.log_params(params)
    y_pred = model.predict_proba(x_test)
    auc = roc_auc_score(y_test, y_pred, multi_class="ovo")
    mlflow.log_metric("auc", auc)
    mlflow.xgboost.log_model(model, artifact_path=model_path)
    mlflow.end_run()


@flow(task_runner=SequentialTaskRunner())
def main() -> None:
    mlflow.set_experiment(conf["mlflow"]["experiment_name"])
    df = load_data(conf["data"]["features"])
    x_train, x_test, y_train, y_test = split_data(
        df,
        conf["model"]["features"],
        conf["model"]["target"],
        conf["model"]["test_size"],
    )
    train_best_model(
        x_train, x_test, y_train, y_test, conf["mlflow"]["model_file_path"]
    )
    return None


if __name__ == "__main__":
    main()
