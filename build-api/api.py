import pickle
import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

# Load the model
with open(os.path.join(os.path.dirname(__file__),  "scorecast_xgboost.pkl"), "rb") as f:
    model = pickle.load(f)


# Input data schema
class PredictionInput(BaseModel):
    features: List[float]


# Define a root `/` endpoint
@app.get('/')
def index():
    return 'Welcome to the Scorecast API!'


# Endpoint to return the prediction
@app.post('/predict')
def predict(input_data: PredictionInput):
    # Convert input data to the format expected by the model
    features = [input_data.features]

    # Predict
    prediction = model.predict(features)

    # Return the prediction as a response
    return {
        'predicted_review_score': float(prediction[0])
        }
