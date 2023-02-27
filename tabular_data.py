# %%
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
# NB, from pandas docs (working with text data): 'There isn’t a clear way to select just text while excluding non-text but still object-dtype columns.'
def set_default_feature_values(dataframe):
    dataframe['beds'].where(~dataframe['beds'].isna(), 1, inplace = True)
    dataframe['guests'].where(~dataframe['guests'].isna(), 1, inplace = True)
    dataframe['guests'] = dataframe['guests'].apply(lambda x: 1 if x == 'Somerford Keynes England United Kingdom' else x)
    dataframe['bathrooms'].where(~dataframe['bathrooms'].isna(), 1, inplace = True)
    dataframe['bedrooms'].where(~dataframe['bedrooms'].isna(), 1, inplace = True)
    dataframe['bedrooms'] = dataframe['bedrooms'].apply(lambda x: 1 if x == 'https://www.airbnb.co.uk/rooms/49009981?adults=1&category_tag=Tag%3A677&children=0&infants=0&search_mode=flex_destinations_search&check_in=2022-04-18&check_out=2022-04-25&previous_page_section_name=1000&federated_search_id=0b044c1c-8d17-4b03-bffb-5de13ff710bc' else x)
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

def load_airbnb(file, feature_columns, label_columns):
    dataset2 = pd.read_csv('airbnb-property-listings/tabular_data/{}'.format(file), index_col= 0)
    features = dataset2[feature_columns]
    labels = dataset2[label_columns]
    return (features,labels)

if __name__ == "__main__":
    dataset = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')
    cleaned_dataset = clean_tabular_data(dataset)
    cleaned_dataset.to_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')

