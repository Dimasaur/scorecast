# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pickle
import os
from datetime import datetime
from xgboost import XGBRegressor

# Load and prepare the data
file_path = "../data/df_restaurants_model.csv"
df_restaurants_model = pd.read_csv(file_path)

# Convert boolean columns to integers
df_restaurants_model = df_restaurants_model.astype({col: 'int' for col in df_restaurants_model.select_dtypes(include=['bool']).columns})

# Feature engineering: create interaction term
df_restaurants_model['delivery_drive_thru'] = df_restaurants_model['delivery'] * df_restaurants_model['drive_thru']
X = df_restaurants_model.drop(columns=['stars'])

# Safely drop columns that may not exist
columns_to_drop = ['appointment_only', 'coat_check', 'drive_thru', 'hours_weekend']
X_reduced = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

y = df_restaurants_model['stars']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocess the data
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Define the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [300],
    'learning_rate': [0.05],
    'max_depth': [7],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
    'gamma': [0.1],
    'reg_alpha': [0],
    'reg_lambda': [10]
}

# Initialize XGBoost Regressor
xgb = XGBRegressor(random_state=42, eval_metric='rmse')

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    cv=5,  # Increased cross-validation for better robustness
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train_processed, y_train)

# Save the best model using pickle
model_filename = "xgb_reg.pkl"
with open(model_filename, 'wb') as file:
    pickle.dump(grid_search.best_estimator_, file)

# Load the model to test if it was saved correctly
with open(model_filename, 'rb') as file:
    xgb_model_loaded = pickle.load(file)

# Test the loaded model
y_pred_loaded = xgb_model_loaded.predict(X_test_processed)
y_pred_original = grid_search.best_estimator_.predict(X_test_processed)

# Verify that predictions match
print("Predictions match:", all(y_pred_loaded == y_pred_original))

# Calculate R-squared
r2 = r2_score(y_test, y_pred_loaded)

# Function to save results to CSV
def save_results_to_csv(model_name, r2_score, best_params, results_csv='model_results.csv'):
    results_df = pd.DataFrame({
        'date_trained': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model_name': [model_name],
        'r2_score': [r2_score],
        'best_params': [str(best_params)]
    })
    results_df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

# Save the results to a CSV file
save_results_to_csv('XGBoostRegressor', r2, grid_search.best_params_)

print(f"Best R-squared: {r2}")
print(f"Best Parameters: {grid_search.best_params_}")


