import streamlit as st
import requests
import json

import numpy as np
import pandas as pd

food_type = pd.read_csv('frontend/food_type.csv')

#import state city dict
state_city_path = '/Users/dima/code/Dimasaur/scorecast/frontend/state_city_dict.json'

with open('frontend/state_city_dict.json') as json_file:
    state_city_dict = json.load(json_file)


with st.form("""## Tell me about the type of restaurant you open ##"""):
    # select the food type you would like to offer
    st.markdown("""### First, select the cuisine type you'd like to have in your restaurants ###""")

    food_type = st.selectbox(
        label = "# Restaurant type #",
        options = food_type['food_type']
    )

    st.markdown("""### Tell us now where you want your restaurant to be located ###""")

    # First dropdown: Select state
    selected_state = st.selectbox(
        "Select a State:",
        options=list(state_city_dict.keys())  # List of states from the dictionary
    )

    # Second dropdown: Select city based on selected state
    if selected_state:
        # Get the list of cities for the selected state
        cities = state_city_dict[selected_state]

        # Create a dropdown for cities
        selected_city = st.selectbox(
            "Select a City:",
            options=list(cities)  # List of cities corresponding to the selected state
        )
    submitted = st.form_submit_button("Submit your preferences")


# API CONNECTIOn

# flavour_forecast_api = "[PLACEHOLDER]" # add the base url here

# display a welcome message from the FastAPI root endpoint

# st.header("Welcome to the API connection")
# response = requests.get(#ADD BASE URL)
# if response.status_code == 200:
#     st.write(response.text)


# # output once the form has been submitted
# if submitted:

#      params = {
#         'food_type' : food_type,
#         'selected_state' : selected_state
#     }

#     response = requests.get(f"{BASE_URL}/predict", params=params)
#     if response.status_code == 200:
#         prediction = response.json()
#         st.write(f"Prediction: {prediction['Av']}")
#     else:
#         st.error("Failed to fetch Flavour Forecast prediction from API.")


# df = pd.DataFrame(
#     [[37.76, -122.4]],
#     columns=["lat", "lon"],
# )

# st.map(df)
