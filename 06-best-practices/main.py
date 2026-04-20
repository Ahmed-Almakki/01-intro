import pickle
import pandas as pd
import requests
from io import StringIO

with open("model.bin", "rb") as f:
    dv, model = pickle.load(f)




def read_data(filename, categorical):
    response = requests.get(filename)
    df = pd.read_parquet(StringIO(response.content.decode('utf-8')))
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year, month):
    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month}.parquet', categorical)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    return y_pred.mean()


if __name__ == "__main__":
    print("Calculating the average predicted duration for the given month and year...")
    year = 2023
    month = 3

    print(f"Processing data for year: {year}, month: {month}...")
    result = main(year, month)
    print(f"Average predicted duration: {result:.2f} minutes")
    output_file = f'taxi_type=yellow_year={year:04d}_month={month:02d}.parquet'
    print(f"Saving results to {output_file}...")
    pd.DataFrame({'predicted_duration': [result]}).to_parquet(output_file, index=False)
    print("Done.")

