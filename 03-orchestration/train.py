import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd 
from prefect import task, flow 
from sklearn.ensemble import RandomForestRegressor 



mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("RandomForestBestParameters")

file_path ="./data/yellow_tripdata_2023-03.parquet"

def load(file: str) -> pd.DataFrame:
    df = pd.read_parquet(file).head(500)
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    df_encoded = pd.get_dummies(df[['PU_DO', 'trip_distance']], columns=['PU_DO'])
    return df_encoded


def train_best(params):
    with mlflow.start_run():
        mlflow.set_tag("developer", "ahmed")
        mlflow.sklearn.autolog()

        model = RandomForestRegressor(**params)
        model.fit(X_train, Y_train)

def register_model():
    """
    Register the model as None and later you can change it to Staging or Proudction
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name("RandomForestBestParameters")
    run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY
        )[0]
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="nyc-taxi-best-model-march-yello")


if __name__ == '__main__':
    df = load(file_path)
    print(f"data frame {df.head(5)}")
    X_train = preprocess(df)
    print("finish preprocessing...")
    Y_train = df['duration'].values
    best_params = {
        "bootstrap": True, "ccp_alpha": 0.0, "criterion": "squared_error", "max_depth": 18, "max_features": 1.0, "max_leaf_nodes": None, "max_samples": None,
        "min_impurity_decrease": 0.0, "min_samples_leaf": 1, "min_samples_split": 10, "min_weight_fraction_leaf": 0.0, "monotonic_cst": None,
        "n_estimators": 38, "n_jobs": None, "oob_score": False, "random_state": 42, "verbose": 0, "warm_start": False
    }
    print("start training")
    train_best(best_params)

    register_model()