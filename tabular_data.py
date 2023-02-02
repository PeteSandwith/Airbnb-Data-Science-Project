# %%
from this import d
from bleach import clean
import pandas as pd
import ast

dataset = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')

def clean_tabular_data(dataframe):
    dataframe = set_default_feature_values(dataframe)
    dataframe = remove_rows_with_missing_ratings(dataframe)
    dataframe = combine_description_strings(dataframe)
    return(dataframe)

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
    dataframe.reset_index(drop=True, inplace=True)
    return dataframe

def combine_description_strings(dataframe):
    #Removes any row without a description
    dataframe.dropna(axis = 0, how = 'any', subset = ['Description'], inplace = True)
    dataframe.reset_index(drop=True, inplace=True)
    #Applies string parsing function to every entry in the Description column
    dataframe['Description'] = dataframe['Description'].apply(parses_description_strings)
    return dataframe


# Function that parses the string-valued entries of the description column, which are strings whose contents are valid lists, and converts them to orderly strings with no erroneous extra spaces.
def parses_description_strings(string):
    try:
        #Uses ast package to parse the string description into a list
        description_as_list = ast.literal_eval(string)
        description_as_list.pop(0)
        #Removes empty quotes from the list
        number_empty_quotes = description_as_list.count('')
        for index in range(0,number_empty_quotes):
            description_as_list.remove('')
        #Turns the list into a single string containing the description
        cleaned_description = ' '.join(description_as_list)
        return cleaned_description
    except: 
        return string

def load_airbnb(file):
    dataset2 = pd.read_csv('airbnb-property-listings/tabular_data/{}'.format(file), index_col= 0)
    features = dataset2[['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']]
    labels = dataset2['Price_Night']
    return (features,labels)

if __name__ == "__main__":
    dataset = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')
    cleaned_dataset = clean_tabular_data(dataset)
    cleaned_dataset.to_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')

