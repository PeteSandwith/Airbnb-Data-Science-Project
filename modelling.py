import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tabular_data import load_airbnb
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import normalize
import joblib
import json
import os

np.random.seed(2)
X, y = load_airbnb('cleaned_tabular_data.csv')
# Normalises the feature data
X = sklearn.preprocessing.normalize(X, norm='l2')
# Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state= 2)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state= 2)

def create_base_model():
    y_predictions_train, y_predictions_test, y_predictions_validation = train_and_predict()
    calculate_R2(y_predictions_train = y_predictions_train, y_predictions_test = y_predictions_test, y_predictions_validation = y_predictions_validation)
    calculate_rmse(y_predictions_train = y_predictions_train, y_predictions_test = y_predictions_test, y_predictions_validation = y_predictions_validation)

# Generates label predictions for the training and test sets
def train_and_predict():
    # Creates and trains the model
    model = linear_model.SGDRegressor()
    model.fit(X_train, y_train)
    
    #Makes predictions
    y_predictions_train = model.predict(X_train)
    y_predictions_test = model.predict(X_test)
    y_predictions_validation = model.predict(X_validation)

    return y_predictions_train, y_predictions_test, y_predictions_validation

# Calculates the R^2 regression score function for training, test and validation sets
def calculate_R2(y_predictions_train, y_predictions_test, y_predictions_validation):
    R2_train = metrics.r2_score(y_train, y_predictions_train)
    R2_test = metrics.r2_score(y_test, y_predictions_test)
    R2_validation = metrics.r2_score(y_validation, y_predictions_validation)
    print('The R2 score for the training set is: ' + str(R2_train))
    print('The R2 score for the test set is: ' + str(R2_test))
    print('The R2 score for the validation set is: ' + str(R2_validation))

# Calculates the root mean squared error for training, test and validation sets
def calculate_rmse(y_predictions_train, y_predictions_test, y_predictions_validation):
    rmse_train = metrics.mean_squared_error(y_train, y_predictions_train, squared = False)
    rmse_test = metrics.mean_squared_error(y_test, y_predictions_test, squared = False)
    rmse_validation = metrics.mean_squared_error(y_validation, y_predictions_validation, squared = False)
    print('The rmse for the training set is: ' + str(rmse_train))
    print('The rmse for the test set is: ' + str(rmse_test))
    print('The rmse for the test set is: ' + str(rmse_validation))

hyperparameters = {'loss': ['epsilon_insensitive', 'squared_error', 'squared_epsilon_insensitive', 'huber'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.00006, 0.00008, 0.0001, 0.00012, 0.00014, 0.00015,  0.00016, 0.00018], 'max_iter': [3000]}

# Custom grid search function 
def custom_tune_regression_model_hyperparameters(model_class, dataset, hyperparameters):
    hyperparameter_combinations = []
    best_model = {'model': 0, 'validation_RMSE': -10000}

    # Fills hyperparameter_combinations with dicitonaries corresponding to all possible combinations of the hyperparameters in the ranges specified.
    for loss in hyperparameters['loss']:
        for penalty in hyperparameters['penalty']:
            for alpha in hyperparameters['alpha']:
                hyperparameter_combinations.append({'loss': loss, 'penalty': penalty, 'alpha': alpha})

    # Trains and evaluates a model for each combination of hyperparameters and determines the best model.
    for combination in hyperparameter_combinations:
        model = model_class(loss = combination['loss'], penalty = combination['penalty'], alpha = combination['alpha'], max_iter= 3000)
        model.fit(dataset['X_train'], dataset['y_train'])
        y_predictions_validation = model.predict(dataset['X_validation'])
        rmse = metrics.mean_squared_error(dataset['y_validation'], y_predictions_validation, squared = False)
        if rmse < best_model['validation_RMSE']:
            best_model['model'] = model
            best_model['validation_RMSE'] = rmse

    return best_model

# Uses GridSearchCV to determine the best comibination of hyperparameters and returns the best performing model. 
def tune_regression_model_hyperparameters(model, hyperparameters):
    grid = sklearn.model_selection.GridSearchCV(estimator= model, param_grid= hyperparameters, scoring= 'r2', refit= 'r2', verbose= 10)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    best_performance_metrics = {'r2': calculate_validation_r2(model = best_estimator), 'rmse': calculate_validation_rmse(model= best_estimator)}
    best_hyperparameters = best_estimator.get_params()

    return best_estimator, best_performance_metrics, best_hyperparameters

def calculate_validation_r2(model):
    y = model.predict(X_validation)
    r2 = metrics.r2_score(y_validation, y)
    return r2

def calculate_validation_rmse(model):
    y = model.predict(X_validation)
    rmse = metrics.mean_squared_error(y_validation, y, squared = False)
    return rmse

def save_model(folder, model, metrics, hyperparameters):
    current_directory = os.getcwd()
    model_filename = folder + 'model.joblib'
    hyperparameters_filename = folder + 'hyperparameters.json'
    performance_metrics_filename = folder + 'metrics.json'

    # Saves the model 
    joblib.dump(model, os.path.join(current_directory, model_filename))

    # Saves the hyperparameters
    with open(os.path.join(current_directory, hyperparameters_filename), "w") as file:
        json.dump(hyperparameters, file) 

    # Saves the metrics
    with open(os.path.join(current_directory, performance_metrics_filename), "w") as file:
        json.dump(metrics, file) 




best_estimator, best_performance_metrics, best_hyperparameters = tune_regression_model_hyperparameters(model = linear_model.SGDRegressor(), hyperparameters= hyperparameters)

#save_model(folder = 'Models/Regression/Linear_Regression/', model = best_estimator, metrics = best_performance_metrics, hyperparameters = best_hyperparameters)



