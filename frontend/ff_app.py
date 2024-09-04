import streamlit as st
import pandas as pd
import json
import time
import pydeck as pdk
from PIL import Image
import plotly.express as px

# Set the page layout to wide
st.set_page_config(layout="wide")

# Load Roboto font and custom CSS for styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

        /* General page styling */
        .main {
            background-color: #f9f9f9;
            padding: 20px;
            font-family: 'Roboto', sans-serif;
        }
        /* Title and subtitle styling */
        h1, h2, h3, h4 {
            color: #1f3864;
            font-family: 'Roboto', sans-serif;
        }
        h2 {
            border-bottom: 2px solid #1f3864;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        h3, h4 {
            font-size: 22px;
            margin-bottom: 5px;
        }
        /* Custom title styling with dark blue background */
        .custom-title {
            background-color: #1f3864;  /* Dark blue background */
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            font-size: 34px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        /* Custom card styling */
        .card {
            background-color: #1f3864;
            border-radius: 10px;
            padding: 30px 20px;
            text-align: center;
            margin-bottom: 20px;
            color: white;
            font-family: 'Roboto', sans-serif;
            height: 150px;
            margin: 10px;
        }
        .card h2 {
            font-size: 36px;
            margin: 0;
            color: white;
        }
        .card p {
            font-size: 16px;
            margin: 0;
            color: white;
        }
        /* Submit button styling */
        .stButton>button {
            background-color: transparent;
            color: #1f3864;
            border: 2px solid #1f3864;
            border-radius: 10px;
            padding: 12px 25px;
            transition: background-color 0.3s, color 0.3s;
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            text-transform: none;
        }
        .stButton>button:hover {
            background-color: #1f3864;
            color: white;
            border: 2px solid #1f3864;
        }
        /* Red button styling for download */
        .stButton.red-button > button {
            background-color: transparent;
            color: #e67c73;
            border: 2px solid #e67c73;
            border-radius: 10px;
            padding: 12px 25px;
            transition: background-color 0.3s, color 0.3s;
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            text-transform: none;
        }
        .stButton.red-button > button:hover {
            background-color: #e67c73;
            color: white;
            border: 2px solid #e67c73;
        }
        /* Ensure no red in button */
        .stButton>button:focus, .stButton>button:active {
            background-color: #1f3864 !important;
            color: white !important;
            border-color: #1f3864 !important;
        }
        /* Selector styling */
        div[data-baseweb="select"] > div {
            background-color: white !important;
            color: #1f3864 !important;
            border-radius: 10px;
            padding: 12px !important;
            font-family: 'Roboto', sans-serif;
            font-size: 16px;
            border: 1px solid #1f3864 !important;
            height: auto !important;
        }
        div[data-baseweb="select"] > div > div {
            overflow: visible !important;
            white-space: nowrap !important;
            text-overflow: ellipsis !important;
        }
        /* Map styling */
        .map-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            height: 100%;
        }
        iframe {
            border: none !important;
            width: 100% !important;
            height: 500px !important;
            border-radius: 10px;
            box-shadow: none;
        }
        /* Centered title with underline */
        .centered-title {
            text-align: center;
            font-size: 34px;  /* Made bigger */
            color: #1f3864;
            border-bottom: 2px solid #1f3864;
            padding-bottom: 10px;
            margin-top: 20px;
            margin-bottom: 20px;
            font-weight: bold;
        }
        /* Success score styling */
        .success-score-container {
            display: flex;
            align-items: flex-start;
            margin-top: 40px;
            width: 100%;
        }
        .success-score-text {
            font-size: 28px;  /* Made bigger */
            color: #1f3864;
            font-family: 'Roboto', sans-serif;
            font-weight: bold;
            margin-bottom: 10px;
            margin-right: 20px;
            flex-grow: 1;
        }
        .success-score-bar {
            height: 40px;
            border-radius: 5px;
            background-color: #e0e0e0;
            width: 100%;
            overflow: hidden;
        }
        .success-score-bar-inner {
            height: 100%;
            border-radius: 5px;
        }
        .success-score-description {
            font-size: 14px;
            color: #1f3864;
            font-family: 'Roboto', sans-serif;
            margin-top: 20px;
        }
        /* Larger and underlined text for competitors */
        .competitors-title {
            font-size: 26px;
            font-weight: bold;
            color: #1f3864;
            margin-top: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f3864;
        }
        /* Larger and underlined text for what top restaurants are doing */
        .top-restaurants-title {
            font-size: 26px;
            font-weight: bold;
            color: #1f3864;
            margin-top: 30px;
            padding-bottom: 10px;
            border-bottom: 2px solid #1f3864;
        }
        /* Loading message styling */
        .loading-message {
            font-size: 18px;
            color: #1f3864;
            font-family: 'Roboto', sans-serif;
            margin-top: 10px;
            display: flex;
            align-items: center;
        }
        .loading-icon {
            width: 50px;
            height: 50px;
            vertical-align: middle;
            margin-left: 10px;
        }
        /* Scroll down text */
        .scroll-down {
            font-size: 20px;  /* Made bigger */
            color: #1f3864;
            font-family: 'Roboto', sans-serif;
            text-align: center;
            margin-top: 10px;  /* Moved up closer to the title */
        }
        .scroll-down-arrow {
            font-size: 24px;  /* Made bigger */
            color: #1f3864;
        }
    </style>
""", unsafe_allow_html=True)

# App title and logo in the same row
col1, col2 = st.columns([4, 1])
with col1:
    st.title("Welcome to Flavor Forecast")
with col2:
    st.image("frontend/logo.png", width=200)  # Made logo bigger

# Subtitle with color change after loading
st.subheader("Make your dream restaurant a reality")

# Open and display the image
image = Image.open("frontend/restaurant1.jpg")
st.image(image, use_column_width=True)

# Load data
food_type_df = pd.read_csv('frontend/food_type.csv')

with open('frontend/state_city_dict.json') as json_file:
    state_city_dict = json.load(json_file)

# Input section in a single row
st.markdown("### Tell us a bit about the restaurant you're dreaming of ###", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    selected_food_type = st.selectbox("Restaurant type", options=food_type_df['food_type'])

with col2:
    selected_state = st.selectbox("Select a State:", options=[state.title() for state in state_city_dict.keys()])

if selected_state:
    cities = state_city_dict[selected_state.upper()]  # Convert back to uppercase to match keys
    with col3:
        selected_city = st.selectbox("Select a City:", options=[city.title() for city in cities])

# Initialize session state for submit button
if "submitted" not in st.session_state:
    st.session_state["submitted"] = False

# Submit button
if st.button("Get success results now"):
    st.session_state["submitted"] = True  # Update session state when button is clicked

# Change the subtitle color to green when loaded
if st.session_state["submitted"]:
    st.markdown("<style>h1 {color: #58bc8b;}</style>", unsafe_allow_html=True)

# Only show results if the submit button was clicked
if st.session_state["submitted"] and selected_city and selected_food_type:
    # Display custom loading message under the button
    loading_placeholder = st.empty()
    loading_placeholder.markdown("""
        <div class="loading-message">
            Finding the recipe for your restaurant success...
            <img src="https://media.giphy.com/media/ZiSxueU7StBwzsvKxw/giphy.gif" class="loading-icon">
        </div>
    """, unsafe_allow_html=True)

    # Simulate a loading period
    time.sleep(3)

    # Clear the loading message
    loading_placeholder.empty()

    # Scroll down message with down arrow
    st.markdown("""
        <div class="scroll-down">
            Scroll down for your report<br>
            <span class="scroll-down-arrow">&#8595;</span>
        </div>
    """, unsafe_allow_html=True)

    # Enhanced centered title with dark blue background box
    st.markdown(f"""
        <div class='custom-title'>
            Your curated guide to opening a {selected_food_type} restaurant in {selected_city}
        </div>
    """, unsafe_allow_html=True)

    # Display success score with a bar using columns
    success_score = 85  # Example score

    # Determine bar color based on the score
    if success_score >= 80:
        bar_color = '#58bc8b'  # Green
        score_color = '#58bc8b'
    elif 60 <= success_score < 80:
        bar_color = '#fac769'  # Yellow
        score_color = '#fac769'
    else:
        bar_color = '#e67c73'  # Red
        score_color = '#e67c73'

    # Use columns to separate the text and bar
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
            <div style="font-size: 28px; color: {score_color}; font-family: 'Roboto', sans-serif; font-weight: bold; margin-bottom: 10px;">
                Your predicted restaurant success score: {success_score}
            </div>
            <div style="font-size: 14px; color: #1f3864; font-family: 'Roboto', sans-serif; margin-top: 20px;">
                How is this calculated? Your predicted restaurant success score is generated by analyzing historical data from restaurants in your city and chosen cuisine type. We evaluate the success of similar establishments based on Yelp review scores within your selected location and cuisine, providing insights into the potential success of your restaurant.
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="height: 40px; border-radius: 5px; background-color: #e0e0e0; width: 100%; overflow: hidden; margin-top: 20px;">
                <div style="height: 100%; width: {success_score}%; background-color: {bar_color}; border-radius: 5px;"></div>
            </div>
        """, unsafe_allow_html=True)

    # Competitors section
    st.markdown(f"<div class='competitors-title'>Top 10 reviewed {selected_food_type} restaurants in {selected_city}</div>", unsafe_allow_html=True)

    # GETTING TOP-10 SIMILAR RESTAURANTS ON THE MAP
    df_restaurants = pd.read_csv("frontend/restaurants_ohe.csv")

    # Filter restaurants in the same city and food type
    df_filtered = df_restaurants[
        (df_restaurants["city"].astype(str).str.upper() == selected_city.upper()) &
        (df_restaurants[f"food_type_one_{selected_food_type}"] == 1)
    ]

    # Get top 10 cities
    df_sorted = df_filtered.sort_values('stars', ascending=False)
    df_map_input = df_sorted[["name", "latitude", "longitude"]].reset_index(drop=True)
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
        get_radius=100,
        get_color=[255, 0, 0, 160],
        pickable=True,
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

    # New Section for What Top Restaurants Are Doing
    st.markdown(f"<div class='top-restaurants-title'>Key facts about the top reviewed restaurants in {selected_city}</div>", unsafe_allow_html=True)

    # ###################################################
    #        # TOP FEATURES OF THE LOCAL RESTAURANTS
    # ###################################################

    restaurants_eda_df_full = pd.read_csv("frontend/restaurant_eda_df_full.csv", low_memory=False)

    selected_city_df = restaurants_eda_df_full[restaurants_eda_df_full.city.astype(str).str.upper() == selected_city.upper()]

    selected_city_stats = selected_city_df.drop(columns=['Unnamed: 0.1', 'key_0', 'Unnamed: 0', 'index', 'business_id',
                                                         'postal_code', 'latitude', 'longitude']).reset_index().drop(
        columns=['index'])

    bool_col = selected_city_stats.select_dtypes(include="object").drop(columns=["food_type", "price_range", "city", "state"])
    bool_col_name = bool_col.columns

    # Clean up the na/nan values from the boolean columns
    selected_city_stats[bool_col_name] = selected_city_stats[bool_col_name].fillna(False)
    selected_city_stats[bool_col_name] = selected_city_stats[bool_col_name].astype(int)
    selected_city_stats['alcohol'] = selected_city_stats['alcohol'].astype(int)

    selected_city_stats = selected_city_stats.drop(columns=["stars", "review_count", "is_open"])
    int_col = selected_city_stats.select_dtypes(include="int").columns

    top_5_features = pd.DataFrame(selected_city_stats[int_col].sum().sort_values(ascending=False)[:5], columns=["percent_of_places"])
    top_5_features['percent_of_places'] = round(top_5_features['percent_of_places'] / len(selected_city_stats) * 100, 1)

    # Add space before pie charts
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Create pie charts for the top 5 features
    pie_charts = []
    for feature in top_5_features.index:
        # Format the feature name by capitalizing and removing underscores
        formatted_feature = feature.replace('_', ' ').capitalize()

        fig = px.pie(values=[top_5_features.loc[feature, 'percent_of_places'], 100 - top_5_features.loc[feature, 'percent_of_places']],
                     names=['', ''],
                     color_discrete_sequence=['#58bc8b', '#e0e0e0'],
                     title=f'{formatted_feature}: {top_5_features.loc[feature, "percent_of_places"]}%')
        fig.update_traces(textinfo='none', hole=.4)  # Remove labels and %
        fig.update_layout(showlegend=False, title_font_size=24, title_font_color='#1f3864', title_x=0.0, paper_bgcolor='#f9f9f9')  # Left-aligned title, larger font size, white background
        pie_charts.append(fig)

    # Display pie charts in a single row
    chart_cols = st.columns(5)
    for i, chart in enumerate(pie_charts):
        with chart_cols[i]:
            st.plotly_chart(chart, use_container_width=True)

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

    selected_state_abbr = state_full_abb[selected_state.upper()]  # Fix for KeyError
    selected_state_df = restaurants_eda_df_full[restaurants_eda_df_full.state.astype(str).str.upper() == selected_state_abbr]

    restaurants_ohe = pd.read_csv("frontend/restaurants_ohe.csv")

    restaurants_ohe_fil = restaurants_ohe[['business_id','food_type_one']]

    selected_state_df_full = pd.merge(selected_state_df, restaurants_ohe_fil,on=["business_id"], how="left")
    state_ranking_food_type = selected_state_df_full.groupby(selected_state_df_full.food_type_one).count()

    top_state_rest = pd.DataFrame(state_ranking_food_type.sort_values("business_id",ascending=False)['index'])

    top_state_rest['count'] = top_state_rest['index']
    del top_state_rest['index']
    top_state_rest['percentage'] = round(top_state_rest['count']/top_state_rest['count'].sum() * 100,1)
    del top_state_rest['count']

    # Combine the remaining percentages into "Other"
    if len(top_state_rest) > 10:
        top_state_rest.loc['Other'] = top_state_rest.iloc[10:].sum()
        top_state_rest = top_state_rest.iloc[:10]

    # Rename the index to "Cuisine type"
    top_state_rest.index.name = 'Cuisine type'

    # Plot the stacked bar chart with no black background
    st.markdown("<br><br>", unsafe_allow_html=True)
    fig_bar = px.bar(top_state_rest, y='percentage', orientation='v', title=f"Most popular cuisine types in {selected_city}",
                     color_discrete_sequence=['#1f3864', '#58bc8b', '#e67c73', '#fac769'])
    fig_bar.update_traces(texttemplate='%{y:.1f}%', textposition='outside', textfont=dict(size=20, color='#1f3864'))  # Larger font for text and dark blue color
    fig_bar.update_layout(
        title_font_size=24, title_x=0.0, paper_bgcolor='#f9f9f9', plot_bgcolor='#f9f9f9', title_font_color='#1f3864',
        xaxis=dict(showgrid=False, tickfont=dict(size=20, color='#1f3864')),  # Axis labels styling
        yaxis=dict(showgrid=False, title='Percentage', tickfont=dict(size=20, color='#1f3864')),  # Y-axis title and labels
        uniformtext_minsize=12, uniformtext_mode='hide', showlegend=False, height=700  # Increased height
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Add a button to download the page as a PDF using JavaScript
    pdf_button_html = """
    <a href="javascript:window.print()" target="_blank" style="text-decoration:none;">
        <div class="stButton red-button">
            <button>Download as PDF</button>
        </div>
    </a>
    """
    st.markdown(pdf_button_html, unsafe_allow_html=True)
