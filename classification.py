import sklearn
from sklearn import metrics
import numpy as np
from modelling import prepare_data


# Use sklearn to compute the key measures of performance for your classification model. 
# That should include the F1 score, the precision, the recall, and the accuracy for both the training and test sets.

X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data = 'cleaned_tabular_data.csv', feature_columns= ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'], label_columns='Category')

model = sklearn.linear_model.LogisticRegression()
model.fit(X_train, y_train)
print(model.score(X_validation, y_validation))

def calculate_f1(X, y, model):
    y_predicted = model.predict(X)
    f1 = metrics.f1_score(y, y_predicted)
    return f1

def calculate_precision(X, y, model):
    y_predicted = model.predict(X)
    precision = metrics.precision_score(y, y_predicted)
    return precision

