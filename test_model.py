import pandas as pd
import joblib

model = joblib.load('random_forest_model.pkl')

data = pd.read_csv('sample_input.csv')

print("Sample input data:")
print(data.head())

predictions = model.predict(data)

print("\nPredictions:")
print(predictions)
