# %%
import pandas as pd

dataset = pd.read_csv('airbnb-property-listings/tabular_data/listing.csv')



# Function that sets missing values in the 'beds', 'guests' 'bathrooms' and 'bedrooms' columns to 1 and returns the updated dataframe.
def set_default_feature_values(dataframe):
    dataframe['beds'] = dataframe['beds'].where(~dataframe['beds'].isna(), 1)
    dataframe['guests'] = dataframe['guests'].where(~dataframe['guests'].isna(), 1)
    dataframe['bathrooms'] = dataframe['bathrooms'].where(~dataframe['bathrooms'].isna(), 1)
    dataframe['bedrooms'] = dataframe['bedrooms'].where(~dataframe['bedrooms'].isna(), 1)
    return dataframe



dataset = set_default_feature_values(dataset)
print(dataset.head(20))

#'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'


# %%
