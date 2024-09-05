import pickle
import os
import numpy as np
from fastapi import FastAPI


app = FastAPI()


# Define a root `/` endpoint
@app.get('/')
def index():
    return 'Welcome to the Scorecast API!'


# Endpoint to return the prediction
@app.post('/predict')
def predict(input_state):
    with open(os.path.join(os.path.dirname(__file__),  "scorecast_xgboost.pkl"), "rb") as f:
        model = pickle.load(f)

    state_encoding_mapping = {
        'ALBERTA': 7.0,
        'ARIZONA': 3.0,
        'CALIFORNIA': 13.0,
        'DELAWARE': 1.0,
        'FLORIDA': 11.0,
        'IDAHO': 10.0,
        'ILLINOIS': 0.0,
        'INDIANA': 4.0,
        'LOUISIANA': 12.0,
        'MISSOURI': 2.0,
        'NEW JERSEY': 6.0,
        'NEVADA': 9.0,
        'PENNSYLVANIA': 8.0,
        'TENNESSEE': 5.0
    }

    input_state = input_state.upper()
    state = state_encoding_mapping[input_state]

    input_features = np.array([1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2.0, 1, 1, 1, 1, 1, 0.0570, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, state, 1])

    input_features_new = input_features.reshape(1, 41)

    # Make a prediction
    prediction = model.predict(input_features_new)

    prediction_scaled = int(prediction[0]/5*100)

    # Return the prediction as a response
    return prediction_scaled


#print(predict('alberta'))
