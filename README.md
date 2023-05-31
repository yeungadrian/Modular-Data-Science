# Modular-Data-Science
Example of using https://github.com/yeungadrian/Data-Science-Template in anger.

## Tools used in this project
* [MLflow](https://mlflow.org/docs/latest/index.html): Experiment tracking and more
* [OmegaConf](https://omegaconf.readthedocs.io/en/2.3_branch/index.html): Manage configuration
* [Poetry](https://python-poetry.org/docs/basic-usage/): Dependency management
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [Prefect](https://docs.prefect.io/): Orchestration and resillient workflows
* [pytest](https://docs.pytest.org/en/latest/): Write small, readable tests
* [python-dotenv](https://pypi.org/project/python-dotenv/): Local testing of secrets
* [Ray-Tune](https://docs.ray.io/en/latest/tune/index.html): Hyperparameter tuning
* [XGBoost](https://xgboost.readthedocs.io/en/stable/index.html): Gradient boosting library

## Case study
- Sklearn iris dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html

## Pipelines

### Data Ingestion
- Read and write data
- Type data

### Feature Engineering
- Feature creation / selection


### Modelling
- Train test split
- Hyperparameter search
- Experiment tracking

## Caveats
- MLflow integration in ray is in alpha

