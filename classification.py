import sklearn
from sklearn import metrics
import numpy as np
from modelling import prepare_data


# Use sklearn to compute the key measures of performance for your classification model. 
# That should include the F1 score, the precision, the recall, and the accuracy for both the training and test sets.

X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data = 'cleaned_tabular_data.csv', feature_columns= ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'], label_columns='Category')

def calculate_scores(X_train, X_test, y_train, y_test, model):
    f1_train = calculate_f1(X_train, y_train, model)
    f1_test = calculate_f1(X_test, y_test, model)
    precision_train = calculate_precision(X_train, y_train, model)
    precision_test = calculate_precision(X_test, y_test, model)
    recall_train = calculate_recall(X_train, y_train, model)
    recall_test = calculate_recall(X_test, y_test, model)
    accuracy_train = calculate_accuracy(X_train, y_train, model)
    accuracy_test = calculate_accuracy(X_test, y_test, model)
    
    print("The f1 score on the training set is: " + str(f1_train) + " and on the test set is: " + str(f1_test))
    print("The precision score on the training set is: " + str(precision_train) + " and on the test set is: " + str(precision_test))
    print("The recall score on the training set is: " + str(recall_train) + " and on the test set is: " + str(recall_test))
    print("The accuracy score on the training set is: " + str(accuracy_train) + " and on the test set is: " + str(accuracy_test))


def calculate_f1(X, y, model):
    y_predicted = model.predict(X)
    f1 = metrics.f1_score(y, y_predicted, average='macro')
    return f1

def calculate_precision(X, y, model):
    y_predicted = model.predict(X)
    precision = metrics.precision_score(y, y_predicted, average='macro')
    return precision

def calculate_recall(X, y, model):
    y_predicted = model.predict(X)
    recall = metrics.recall_score(y, y_predicted, average='macro')
    return recall

def calculate_accuracy(X, y, model):
    y_predicted = model.predict(X)
    accuracy = metrics.accuracy_score(y, y_predicted)
    return accuracy

model = sklearn.linear_model.LogisticRegression()
model.fit(X_train, y_train)
calculate_scores(X_train, X_test, y_train, y_test, model)