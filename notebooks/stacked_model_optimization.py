
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingRegressor
import joblib
import os
from datetime import datetime

# Load the data
file_path = "../data/df_restaurants_model.csv"
df_restaurants_model = pd.read_csv(file_path)

# Convert boolean columns to integers
df_restaurants_model = df_restaurants_model.astype({col: 'int' for col in df_restaurants_model.select_dtypes(include=['bool']).columns})

# Feature engineering
df_restaurants_model['delivery_drive_thru'] = df_restaurants_model['delivery'] * df_restaurants_model['drive_thru']
X = df_restaurants_model.drop(columns=['stars'])
y = df_restaurants_model['stars']

# Safely drop columns that may not exist
columns_to_drop = ['appointment_only', 'coat_check', 'drive_thru', 'hours_weekend']
X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a preprocessing pipeline
preprocessing = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Apply preprocessing
X_train_processed = preprocessing.fit_transform(X_train)
X_test_processed = preprocessing.transform(X_test)

# Define base models
base_models = [
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=100, random_state=42, eval_metric='rmse'))
]

# Define the meta-model
meta_model = LinearRegression()

# Create the Stacking Regressor
stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Use cross-validation to generate the training set for the meta-model
)

# Train the stacked model
stacking_model.fit(X_train_processed, y_train)

# Predict on the test set
y_pred = stacking_model.predict(X_test_processed)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
print(f"Stacked Model R-squared: {r2}")

# Print out the best parameters of each base model
for name, model in stacking_model.named_estimators_.items():
    print(f"Best parameters for {name}: {model.get_params()}")

# Save the stacked model
model_filename = "stacked_model.joblib"
joblib.dump(stacking_model, model_filename)
print(f"Stacked model saved as {model_filename}")

# Save results to CSV
def save_results_to_csv(model_name, r2_score, model_params, results_csv='model_results.csv'):
    results_df = pd.DataFrame({
        'date_trained': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        'model_name': [model_name],
        'r2_score': [r2_score],
        'model_params': [str(model_params)]
    })
    results_df.to_csv(results_csv, mode='a', header=not os.path.exists(results_csv), index=False)

# Collect all parameters for saving
all_params = {name: model.get_params() for name, model in stacking_model.named_estimators_.items()}
save_results_to_csv('StackedModel', r2, all_params)
