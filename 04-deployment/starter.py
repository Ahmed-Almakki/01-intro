#!/usr/bin/env python
import argparse
import pickle
import pandas as pd


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('year', type=int, help='Year of the data')
parser.add_argument('month', type=int, help='Month of the data')

args = parser.parse_args()
year = args.year
month = args.month
print(f"Processing data for {year:04d}-{month:02d}...")

with open('model.bin', 'rb') as f:
    dv, model = pickle.load(f)


categorical = ['PULocationID', 'DOLocationID']
def read_data(filename):
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df



df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet')

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)

print(f"Standard deviation:{df['duration'].std():.2f}")

df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
df.to_parquet(f'data/yellow_tripdata_{year}-{month:02d}.parquet', engine='pyarrow', compression=None, index=False)

print(f"Predicted mean duration: {y_pred.mean():.2f} minutes")
