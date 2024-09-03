# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime
from xgboost import XGBRegressor
import statsmodels.api as sm

# Load the data
file_path = "../data/df_restaurants_model.csv"
df_restaurants_model = pd.read_csv(file_path)

# Prepare the data
df_restaurants_model = df_restaurants_model.astype({col: 'int' for col in df_restaurants_model.select_dtypes(include=['bool']).columns})

# Feature engineering
df_restaurants_model['delivery_drive_thru'] = df_restaurants_model['delivery'] * df_restaurants_model['drive_thru']
X = df_restaurants_model.drop(columns=['stars'], errors='ignore')
X_reduced = df_restaurants_model.drop(columns=['stars', 'appointment_only', 'coat_check', 'drive_thru', 'hours_weekend'], errors='ignore')
y = df_restaurants_model['stars']  # Target

# Add a constant to the model (intercept)
X_reduced = sm.add_constant(X_reduced)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Create a pipeline with an imputer and standard scaler
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler())  # Standardize features
])

# Fit the pipeline on X_train and transform both X_train and X_test
X_train_processed = pipeline.fit_transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Define a smaller parameter grid for XGBoost
param_grid = {
    'n_estimators': [100, 200, 300],  # Number of boosted trees to fit
    'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinkage
    'max_depth': [3, 5, 7],  # Maximum tree depth
    'subsample': [0.8, 1.0],  # Subsample ratio of the training instance
    'colsample_bytree': [0.8, 1.0],  # Subsample ratio of columns when constructing each tree
    'gamma': [0, 0.1],  # Minimum loss reduction required to make a further partition
    'reg_alpha': [0, 0.1],  # L1 regularization term on weights
    'reg_lambda': [1, 10]  # L2 regularization term on weights
}

# Initialize XGBoost Regressor
xgb = XGBRegressor(random_state=42, early_stopping_rounds=10, eval_metric='r2')

# Initialize GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='r2',
    cv=3,  # Reduced to 3 folds
    n_jobs=-1,
    verbose=1
)

# Perform Grid Search
grid_search.fit(X_train_processed, y_train, eval_set=[(X_test_processed, y_test)], verbose=False)

# Get the best model and parameters
best_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict on the test set
y_pred = best_model.predict(X_test_processed)
r2 = r2_score(y_test, y_pred)

# Save the model to a file
model_filename = "xgboost_best_model.joblib"
joblib.dump(best_model, model_filename)
print(f"Best model saved as {model_filename}")

# Function to save results and best parameters to CSV
def save_results_to_csv(model_name, r2_score, best_params, results_csv='model_results.csv'):
    date_trained = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    csv_filename = f"{model_name}.csv"

    # Create DataFrame to save results
    results_df = pd.DataFrame({
        'date_trained': [date_trained],
        'csv_filename': [csv_filename],
        'r2_score': [r2_score],
        'best_params': [str(best_params)]
    })

    # Append results to CSV
    if os.path.exists(results_csv):
        results_df.to_csv(results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv, index=False)

    print(f"Results and parameters appended to {results_csv}")

# Save the results
save_results_to_csv('XGBoostRegressor', r2, best_params)

print(f"Best R-squared: {r2}")
print(f"Best Parameters: {best_params}")
