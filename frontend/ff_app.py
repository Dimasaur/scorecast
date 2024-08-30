import streamlit as st

import numpy as np
import pandas as pd
from food_type import df_food_t20
from cities_states import states, cities, state_city_dict

cities_list = list(cities['city'])
states_list = list(states['state'])

st.markdown("""## Tell me about the type of restaurant you open ##""")

# select the food type you would like to offer
st.markdown("""## First, select the cuisine type you'd like to have in your restaurants ##""")

st.selectbox(
    label = "# Restaurant type #",
    options = df_food_t20
)


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

# st.selectbox(
#     label = "# State #",
#     options = list(states_city_dict.keys())
# )

# st.selectbox(
#     label = "# City #",
#     options = cities_list
# )
# select the food type you would like to offer
