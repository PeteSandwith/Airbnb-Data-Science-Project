# %%
import pandas as pd

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


dataset = set_default_feature_values(dataset)
dataset = remove_rows_with_missing_ratings(dataset)
print(dataset.head(20))



# %%
