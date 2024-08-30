import pandas as pd
import numpy as np

# restaurants_limited_features
df_res = pd.read_csv("/Users/dima/code/Dimasaur/scorecast/data/restaurants_limited_features.csv", low_memory=False)

df_res['food_type_string'] = df_res['food_type'].apply(lambda x: str(x).strip('[]'))
df_res['food_type_string'] = df_res['food_type_string'].apply(lambda x: str(x).replace("'",""))

# replace the values in the food_types with the top - 20 cuisines

replace_dictionary = {
    "Pizza" : "Italian",
    "Burgers, Fast Food" : "Burgers",
    "Italian, Pizza" : "Italian",
    "American (New)" : "American",
    "Fast Food, Sandwiches" : "Sandwiches",
    "American (Traditional)" : "American",
    "Fast Food, Mexican" : "Mexican",
    "Coffee & Tea, Food" : "Cafes",
    "Japanese, Sushi Bars" : "Japanese",
    "American (Traditional), Diners" : "Diners",
    "Bakeries, Food" : "Bakery",
    "Delis, Sandwiches" : "Delis",
    "Mexican, Tex-Mex" : "Mexican",
    "American (Traditional), Burgers" : "American",
    "Sushi Bars" : "Japanese",
    "Chinese, Fast Food" : "Chinese",
    "American (Traditional), Cafes" : "American",
    "Mexican, Tacos" : "Mexican",
    "Pizza, Sandwiches" : "Sandwiches",
    "Food, Pizza" : "Italian",
    "American (Traditional), Seafood" : "American",
    "Greek, Mediterranean" : "Greek",
    "American (New), American (Traditional)" : "American",
    "Cajun/Creole, Seafood" : "Cajun/Creole"

}

df_res.food_type_string = df_res.food_type_string.replace(replace_dictionary)

# generate the clea array of the top 20 categories
df_food_t20 = list(df_res.food_type_string.value_counts(normalize=True).head(25).index)
df_food_t20.remove('O')
df_food_t20.remove('')
df_food_t20 = df_food_t20[:20]

# top 20 food types
np_food_20 = np.array([df_food_t20])
