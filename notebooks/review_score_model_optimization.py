# -*- coding: utf-8 -*-
"""review_score_model_optimization.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11ieBgCUyRB5MDPt_Mn_T_6kCruwxuMAM
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from google.cloud import storage
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR

# Creating a destination path
file_path = "/content/df_restaurants_model.csv"

df_restaurants_model = pd.read_csv(file_path)

# Step 1: Prepare the data
# Ensure all boolean columns are numeric (True/False -> 1/0)
df_restaurants_model = df_restaurants_model.astype({col: 'int' for col in df_restaurants_model.select_dtypes(include=['bool']).columns})

# Step 2: Split the data into features (X) and target (y)
# Creating new terms
df_restaurants_model['delivery_drive_thru'] = df_restaurants_model['delivery'] * df_restaurants_model['drive_thru']
X = df_restaurants_model.drop(columns=['stars'], errors='ignore')
X_reduced = df_restaurants_model.drop(columns=['stars', 'appointment_only', 'coat_check', 'drive_thru', 'hours_weekend'], errors='ignore')
y = df_restaurants_model['stars']  # Target

# Add a constant to the model (intercept)
X_reduced = sm.add_constant(X_reduced)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

df_restaurants_model.head()

"""# 2 Testing different models

*   Use SGDRegressor and perform GridSearchCV to fine-tune ‘penalty’ and ‘alpha’
*   Implement KNeighborsRegressor with GridSearchCV to optimize hyperparameters
*   Evaluate the model using RandomForestRegressor
*   Test GradientBoostingRegressor for potential enhancements
*   Explore XGBoostRegressor as a more advanced boosting method
*   Experiment with SVR and try different kernel options
"""

# Set GCP bucket name
BUCKET_NAME = 'flavor-forecast'
MODEL_PATH = 'models/flavor-forecast-models'

# Initialize Google Cloud Storage client
client = storage.Client()

# Function to save model to GCS
def save_model_to_gcs(model, model_name):
    model_filename = f"{model_name}.joblib"
    joblib.dump(model, model_filename)

    gcs_model_path = f"{MODEL_PATH}{model_filename}"

    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_model_path)
    blob.upload_from_filename(model_filename)

    os.remove(model_filename)
    print(f"Model saved to GCS bucket at {gcs_model_path}")

# Function to append results to CSV
def append_results_to_csv(model_name, r_squared, results_csv='model_results.csv'):
    date_trained = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_filename = f"{model_name}.csv"

    if os.path.exists(results_csv):
        df_results = pd.read_csv(results_csv)
    else:
        df_results = pd.DataFrame(columns=['date_trained', 'csv_filename', 'r_squared'])

    new_row = {'date_trained': date_trained, 'csv_filename': csv_filename, 'r_squared': r_squared}
    df_results = df_results.append(new_row, ignore_index=True)

    df_results.to_csv(results_csv, index=False)
    print(f"Results appended to {results_csv}")

# Define parameter grids for each model
model_configs = {
    'SGDRegressor': {
        'model': SGDRegressor(random_state=42),
        'params': {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'penalty': ['l2', 'l1', 'elasticnet'],
            'max_iter': [1000, 2000, 5000],
            'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
            'eta0': [0.01, 0.1, 1.0]
        }
    },
    'KNeighborsRegressor': {
        'model': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 10],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        }
    },
    'RandomForestRegressor': {
        'model': RandomForestRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 500],
            'max_features': ['sqrt', 'log2'],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
    },
    'GradientBoostingRegressor': {
        'model': GradientBoostingRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
        }
    },
    'XGBoostRegressor': {
        'model': XGBRegressor(random_state=42),
        'params': {
            'n_estimators': [100, 200, 500],
            'learning_rate': [0.01, 0.1, 0.05],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
        }
    },
    'SVMRegressor': {
        'model': SVR(),
        'params': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        }
    }
}

best_overall_model = None
best_r2_score = float('-inf')
best_model_name = ""

for model_name, config in model_configs.items():
    print(f"Training {model_name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    Save the model to GCS
    save_model_to_gcs(best_model, model_name)

    Append results to the CSV
    append_results_to_csv(model_name, r2)

    # Track the best model overall
    if r2 > best_r2_score:
        best_r2_score = r2
        best_overall_model = best_model
        best_model_name = model_name

print(f"Best overall model: {best_model_name} with R-squared: {best_r2_score}")