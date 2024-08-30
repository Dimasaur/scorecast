import streamlit as st
import requests
import json

import numpy as np
import pandas as pd

food_type = pd.read_csv('/Users/dima/code/Dimasaur/scorecast/frontend/food_type.csv')

#import state city dict
state_city_path = '/Users/dima/code/Dimasaur/scorecast/frontend/state_city_dict.json'

with open('state_city_dict.json') as json_file:
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
            options=cities  # List of cities corresponding to the selected state
        )
    submitted = st.form_submit_button("Submit your preferences")

params = {
    'food_type' : food_type,
    'selected_state' : selected_state,
    'selected_city' : selected_city
}

#flavour_forecast_api = "[PLACEHOLDER]"

#response = requests.get(flavour_forecast_api,params=params)

#prediction = response.json()
#predicted_score = prediction['prediction score']
