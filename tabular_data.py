# %%
from this import d
import pandas as pd
import ast

dataset = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')



# Function that sets missing values in the 'beds', 'guests' 'bathrooms' and 'bedrooms' columns to 1 and returns the updated dataframe.
def set_default_feature_values(dataframe):
    dataframe['beds'].where(~dataframe['beds'].isna(), 1, inplace = True)
    dataframe['guests'].where(~dataframe['guests'].isna(), 1, inplace = True)
    dataframe['bathrooms'].where(~dataframe['bathrooms'].isna(), 1, inplace = True)
    dataframe['bedrooms'].where(~dataframe['bedrooms'].isna(), 1, inplace = True)
    return dataframe

# Function that removes any rows that have missing values in the ratings columns
def remove_rows_with_missing_ratings(dataframe):
    dataframe.dropna(axis = 0, how = 'any', subset = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace = True)
    return dataframe

#def combine_description_strings(dataframe):



dataset = set_default_feature_values(dataset)
dataset = remove_rows_with_missing_ratings(dataset)


def parses_description_strings(string):
    #Uses ast package to parse the string description into a list
    description_as_list = ast.literal_eval(string)
    #Removes empty quotes from the list
    number_empty_quotes = description_as_list.count('')
    for index in range(0,number_empty_quotes):
        description_as_list.remove('')
    #Turns the list into a single string containing the description
    cleaned_description = ' '.join(description_as_list)
    return cleaned_description




# %%
