import sklearn
import numpy as np
from modelling import prepare_data


X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data = 'cleaned_tabular_data.csv', feature_columns= ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'], label_columns='Category')

