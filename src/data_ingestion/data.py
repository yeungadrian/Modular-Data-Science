import pandas as pd
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from sklearn.datasets import load_iris

@task
def load_data() -> pd.DataFrame:
    df = load_iris(as_frame=True)['frame']
    return df

@task
def write_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path)
    return None

@flow(task_runner=SequentialTaskRunner())
def main() -> None:
    df = load_data()
    write_data(df, 'data/interim/iris.csv')

if __name__ == "__main__":
    main()