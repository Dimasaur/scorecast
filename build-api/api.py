from fastapi import FastAPI

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return 'Welcome to the Scorecast API!'

@app.get('/predict')
def predict():
    return {'Av': 64}
