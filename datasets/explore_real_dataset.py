import pandas as pd

import app  # noqa: F401 — ensure project root on sys.path
from app.paths import ECG_FEATURES_CSV

df = pd.read_csv(
    ECG_FEATURES_CSV
)

print(df.head())

print("\nShape:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nStatistics:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())