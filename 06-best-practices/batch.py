from io import BytesIO
import os
import pandas as pd
import pickle
from pathlib import Path
import requests


BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model" / "model.bin", "rb") as f:
    dv, model = pickle.load(f)


def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)


def get_output_path(year, month):
    default_output_pattern = 's3://nyc-duration-prediction-alexey/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)


def read_data(filename):
    s3_endpoint_url = os.getenv('S3_ENDPOINT_URL')
    if s3_endpoint_url is not None:
        options = {'client_kwargs': {'endpoint_url': s3_endpoint_url}}
        return pd.read_parquet(filename, storage_options=options)
    response = requests.get(filename)
    response.raise_for_status()
    df = pd.read_parquet(BytesIO(response.content))
    return df

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)

    df = read_data(input_file)
    df = prepare_data(df, categorical)
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    pd.DataFrame({'predicted_duration': y_pred}).to_parquet(output_file, index=False)

    return y_pred.mean()


if __name__ == "__main__":
    print("Calculating the average predicted duration for the given month and year...")
    year = 2023
    month = 3

    print(f"Processing data for year: {year}, month: {month}...")
    result = main(year, month)
    print(f"Average predicted duration: {result:.2f} minutes")
