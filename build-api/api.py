import pickle
import os
from fastapi import FastAPI

app = FastAPI()

# Load the model at startup
with open(os.path.join(os.path.dirname(__file__),  "scorecast_xgboost.pkl"), "rb") as f:
    model = pickle.load(f)



# Define a root `/` endpoint
@app.get('/')
def index():
    return 'Welcome to the Scorecast API!'


@app.get('/predict')
def predict():
    return {'Predicted review score': 3.9}
