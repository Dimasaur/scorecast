import pandas as pd
import numpy as np
import json

cities = pd.read_csv('cities.csv')
states = pd.read_csv('states.csv')

#import state city dict
state_city_path = '/Users/dima/code/Dimasaur/scorecast/frontend/state_city_dict.json'

with open('state_city_dict.json') as json_file:
    state_city_dict = json.load(json_file)
