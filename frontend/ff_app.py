import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import pydeck as pdk

food_type = pd.read_csv('frontend/food_type.csv')

#import state city dict
state_city_path = '/Users/dima/code/Dimasaur/scorecast/frontend/state_city_dict.json'

with open('frontend/state_city_dict.json') as json_file:
    state_city_dict = json.load(json_file)

st.markdown("""### First, select the cuisine type you'd like to have in your restaurants ###""")

food_type = st.selectbox(
    label = "# Restaurant type #",
    options = food_type['food_type']
    )

st.markdown("""### Tell us now where you want your restaurant to be located ###""")

# First dropdown: Select state
selected_state = st.selectbox(
    "Select a State:",
    options=list(state_city_dict.keys()))  # List of states from the dictionary

    # Second dropdown: Select city based on selected state
if selected_state:
    # Get the list of cities for the selected state
    cities = state_city_dict[selected_state]

    # Create a dropdown for cities
    selected_city = st.selectbox(
        "Select a City:",
        options=list(cities)  # List of cities corresponding to the selected state
        )

submitted = st.button("Submit your preferences")


# API CONNECTIOn

flavour_forecast_api = "https://scorecast-260513249034.europe-west1.run.app/" # add the base url here

# display a welcome message from the FastAPI root endpoint

st.header("Welcome to the API connection")
response = requests.get(flavour_forecast_api)
if response.status_code == 200:
    st.write(response.text)


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

# search for the city's best restaurants


# GETTING TOP-1O SIMILAR RESTAURANTS ON THE MAP
df_restaurants = pd.read_csv("frontend/restaurants_ohe.csv")

# filter the of the same city and food type
df_filtered = df_restaurants[
    (df_restaurants["city"].astype(str).str.upper() == selected_city) &
    (df_restaurants[f"food_type_one_{food_type}"] == 1)
]

# get top 10 cities
df_sorted = df_filtered.sort_values('stars',ascending=False)
df_map_input = df_sorted[["name","latitude","longitude"]].reset_index()
del df_map_input['index']
df_map_input = df_map_input[:10]




# Define the initial view for the map
initial_view = pdk.ViewState(
    latitude=df_map_input['latitude'].mean(),
    longitude=df_map_input['longitude'].mean(),
    zoom=12,
    pitch=0
)

# Define the scatterplot layer
layer = pdk.Layer(
    'ScatterplotLayer',
    data=df_map_input,
    get_position='[longitude, latitude]',
    get_radius=200,  # Radius of the circles
    get_color=[255, 0, 0, 160],  # Color of the circles (RGBA)
    pickable=True,  # Allows picking the circles to show tooltips
)

# Define the tooltip to display restaurant names
tooltip = {
    "html": "<b>Restaurant Name:</b> {name}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}

# Create a Pydeck map with the defined layer and tooltip
deck = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
    initial_view_state=initial_view,
    layers=[layer],
    tooltip=tooltip
)

# Display the map in Streamlit
st.pydeck_chart(deck)


# st.map(data=df_map_input,
#        latitude=df_map_input.latitude,
#        longitude=df_map_input.longitude,
#        size=100)
