import sklearn
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from modelling import prepare_data


# Use sklearn to compute the key measures of performance for your classification model. 
# That should include the F1 score, the precision, the recall, and the accuracy for both the training and test sets.


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

def tune_classification_model_hyperparameters(model, hyperparameters):
    f1_scorer = metrics.make_scorer(metrics.f1_score, average= "macro")
    grid = sklearn.model_selection.GridSearchCV(estimator= model, param_grid= hyperparameters, scoring= f1_scorer, verbose= 10)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    best_performance_metrics = {'f1': calculate_f1(X_train, y_train, model = best_estimator)}
    best_hyperparameters = best_estimator.get_params()

    return best_estimator, best_performance_metrics, best_hyperparameters


if __name__ == "__main__":

    logistic_regression_hyperparameters = {
        'penalty': ['l2', None],
        'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        'max_iter': [200],
        'C': [0.8, 0.9, 1.0, 1.1, 1.2],

        }

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data = 'cleaned_tabular_data.csv', feature_columns= ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'], label_columns='Category')

    best_estimator, best_performance_metrics, best_hyperparameters = tune_classification_model_hyperparameters(linear_model.LogisticRegression(), logistic_regression_hyperparameters)
    print(best_estimator)
    print(best_hyperparameters)
    print(best_performance_metrics)


    #model = sklearn.linear_model.LogisticRegression()
    #model.fit(X_train, y_train)
    #calculate_scores(X_train, X_test, y_train, y_test, model)