import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@task
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(
        columns={
            "sepal length (cm)": "sepal_length",
            "sepal width (cm)": "sepal_width",
            "petal length (cm)": "petal_length",
            "petal width (cm)": "petal_width",
            "target": "target",
        }
    )
    return df


@task
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    return df


@task
def write_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    return None


@flow(task_runner=SequentialTaskRunner())
def main() -> None:
    df = load_data("data/interim/iris.csv")
    df = rename_columns(df)
    df = add_features(df)
    write_data(df, "data/processed/iris.csv")
    return None


if __name__ == "__main__":
    main()
