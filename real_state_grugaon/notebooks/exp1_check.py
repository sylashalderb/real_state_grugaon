import numpy as np
import pandas as pd
import re
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess data
df = pd.read_csv('flats.csv')
df.drop(columns=['link', 'property_id'], inplace=True)
df.rename(columns={'area': 'price_per_sqft'}, inplace=True)

# Clean society names
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()

# Clean price column
df = df[df['price'] != 'Price on Request']

def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0]) / 100, 2)
        else:
            return round(float(x[0]), 2)

df['price'] = df['price'].str.split(' ').apply(treat_price)
df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹', '').str.replace(',', '').str.strip().astype('float')
df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')
df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')
df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No', '0').astype('int')
df['additionalRoom'].fillna('not available', inplace=True)
df['additionalRoom'] = df['additionalRoom'].str.lower()
df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground', '0').str.replace('Basement', '-1').str.replace('Lower', '0').str.extract(r'(\d+)').astype('float')
df['facing'].fillna('NA', inplace=True)

# Feature engineering
df.insert(loc=4, column='area', value=round((df['price'] * 10000000) / df['price_per_sqft']))
df.insert(loc=1, column='property_type', value='flat')

# Save cleaned data
df.to_csv('flats_cleaned.csv', index=False)

# Set up MLflow and DagsHub
mlflow.set_tracking_uri("https://dagshub.com/sylashalderb/project_kalke.mlflow")
dagshub.init(repo_owner='sylashalderb', repo_name='project_kalke', mlflow=True)
mlflow.set_experiment("Price Prediction Model")

with mlflow.start_run():
    # Log preprocessing parameters
    mlflow.log_param("data_source", "flats.csv")
    mlflow.log_param("features_used", "price, price_per_sqft, bedRoom, bathroom, balcony, additionalRoom, floorNum, facing, area")

    # Prepare data for training
    X = df[['price_per_sqft', 'bedRoom', 'bathroom', 'balcony', 'additionalRoom', 'floorNum']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Log model parameters
    mlflow.log_param("model", "Linear Regression")

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("mean_absolute_error", mae)
    mlflow.log_metric("mean_squared_error", mse)
    mlflow.log_metric("r2_score", r2)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Print results
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")