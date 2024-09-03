import streamlit as st
import requests
import json
import numpy as np
import pandas as pd
import pydeck as pdk


###################################################
            # FRONTEND BASIC OPTIONS
###################################################

food_type = pd.read_csv('frontend/food_type.csv')

#import state city dict
state_city_path = '/Users/dima/code/Dimasaur/scorecast/frontend/state_city_dict.json'

with open('frontend/state_city_dict.json') as json_file:
    state_city_dict = json.load(json_file)

st.markdown("""### First, select the cuisine type you'd like to have in your restaurants ###""")

food_type = st.selectbox(
    label = "Restaurant type",
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

###################################################
                # API CONNECTION
###################################################

flavour_forecast_api = "https://scorecast-260513249034.europe-west1.run.app/" # add the base url here

# display a welcome message from the FastAPI root endpoint
st.header("Welcome to the API connection")
response = requests.get(flavour_forecast_api)
if response.status_code == 200:
    st.write(response.text)

# output once the form has been submitted
if submitted:
    params = {
        'food_type' : food_type,
        'selected_state' : selected_state
    }
    response = requests.get(f"{flavour_forecast_api}/predict", params=params)

    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prediction: {prediction}")
    else:
        st.error("Failed to fetch Flavour Forecast prediction from API.")

###################################################
  # GETTING TOP-1O SIMILAR RESTAURANTS ON THE MAP
###################################################

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
    get_radius=100,  # Radius of the circles
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

# ###################################################
#        # TOP FEATURES OF THE LOCAL RESTAURANTS
# ###################################################

restaurants_eda_df_full = pd.read_csv("frontend/restaurant_eda_df_full.csv",low_memory=False)

selected_city_df = restaurants_eda_df_full[restaurants_eda_df_full.city.astype(str).str.upper() == selected_city]

selected_city_stats = selected_city_df.drop(columns = ['Unnamed: 0.1', 'key_0', 'Unnamed: 0', 'index', 'business_id',
       'postal_code', 'latitude', 'longitude']).reset_index().drop(columns=['index'])

bool_col = selected_city_stats.select_dtypes(include="object").drop(columns=["food_type","price_range","city","state"])
bool_col_name = bool_col.columns
# clean up the na/nan values from the boolean columns

selected_city_stats[bool_col_name] = selected_city_stats[bool_col_name].fillna(False)
selected_city_stats[bool_col_name].isna().sum()

selected_city_stats[bool_col_name] = selected_city_stats[bool_col_name].astype(int)
selected_city_stats['alcohol'] = selected_city_stats['alcohol'].astype(int)

selected_city_stats = selected_city_stats.drop(columns = ["stars","review_count","is_open"])
int_col = selected_city_stats.select_dtypes(include="int").columns

top_5_features = pd.DataFrame(selected_city_stats[int_col].sum().sort_values(ascending = False)[:5], columns = ["percent_of_places"])

top_5_features = pd.DataFrame(round(top_5_features.percent_of_places / len(selected_city_stats),2)*100)

st.dataframe(top_5_features)


# ###################################################
#  # MOST POPULAR FOOD TYPES IN THE SELECTED STATE
# ###################################################

state_ab_full = {
    'AB': 'ALBERTA',
    'AZ': 'ARIZONA',
    'CA': 'CALIFORNIA',
    'DE': 'DELAWARE',
    'FL': 'FLORIDA',
    'ID': 'IDAHO',
    'IL': 'ILLINOIS',
    'IN': 'INDIANA',
    'LA': 'LOUISIANA',
    'MO': 'MISSOURI',
    'NJ': 'NEW JERSEY',
    'NV': 'NEVADA',
    'PA': 'PENNSYLVANIA',
    'TN': 'TENNESSEE'
}

state_full_abb = {
    'ALBERTA': 'AB',
    'ARIZONA': 'AZ',
    'CALIFORNIA':'CA',
    'DELAWARE':'DE',
    'FLORIDA':'FL',
    'IDAHO':'ID',
    'ILLINOIS':'IL',
    'INDIANA': 'IN',
    'LOUISIANA':'LA',
    'MISSOURI': 'MO',
    'NEW JERSEY':'NJ',
    'NEVADA': 'NV',
    'PENNSYLVANIA':'PA',
    'TENNESSEE':'TN'
}

selected_state_abbr = state_full_abb[selected_state]
selected_state_df = restaurants_eda_df_full[restaurants_eda_df_full.state.astype(str).str.upper() == selected_state_abbr]

restaurants_ohe = pd.read_csv("/Users/dima/code/Dimasaur/scorecast/data/restaurants_ohe.csv")

restaurants_ohe_fil = restaurants_ohe[['business_id','food_type_one']]

selected_state_df_full = pd.merge(selected_state_df, restaurants_ohe_fil,on=["business_id"], how="left")
state_ranking_food_type = selected_state_df_full.groupby(selected_state_df_full.food_type_one).count()

top_state_rest = pd.DataFrame(state_ranking_food_type.sort_values("business_id",ascending=False)['index'])

top_state_rest['count'] = top_state_rest['index']
del top_state_rest['index']
top_state_rest['percentage'] = round(top_state_rest['count']/top_state_rest['count'].sum() * 100,0)
del top_state_rest['count']
top_state_rest = top_state_rest[:10]

st.dataframe(top_state_rest)
