from multiprocessing.spawn import prepare
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from tabular_data import load_airbnb
from sklearn import linear_model
from sklearn import metrics
from sklearn import tree
from sklearn import ensemble
import numpy as np
from sklearn.preprocessing import normalize
import joblib
import json
import os

def prepare_data(data):
    np.random.seed(2)
    X, y = load_airbnb(data)
    # Normalises the feature data
    X = sklearn.preprocessing.normalize(X, norm='l2')
    # Splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state= 2)
    X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state= 2)
    return X_train, X_validation, X_test, y_train, y_validation, y_test

class base_regression_model:
    def __init__(self):
        y_predictions_train, y_predictions_test, y_predictions_validation = self.__train_and_predict__()
        self.__calculate_R2__(y_predictions_train = y_predictions_train, y_predictions_test = y_predictions_test, y_predictions_validation = y_predictions_validation)
        self.__calculate_rmse__(y_predictions_train = y_predictions_train, y_predictions_test = y_predictions_test, y_predictions_validation = y_predictions_validation)

    
    def __train_and_predict__(self):
        # Creates and trains the model
        model = linear_model.SGDRegressor()
        model.fit(X_train, y_train)
        
        #Makes predictions
        y_predictions_train = model.predict(X_train)
        y_predictions_test = model.predict(X_test)
        y_predictions_validation = model.predict(X_validation)

        return y_predictions_train, y_predictions_test, y_predictions_validation

    # Calculates the R^2 regression score function for training, test and validation sets
    def __calculate_R2__(self, y_predictions_train, y_predictions_test, y_predictions_validation):
        R2_train = metrics.r2_score(y_train, y_predictions_train)
        R2_test = metrics.r2_score(y_test, y_predictions_test)
        R2_validation = metrics.r2_score(y_validation, y_predictions_validation)
        print('The R2 score for the training set is: ' + str(R2_train))
        print('The R2 score for the test set is: ' + str(R2_test))
        print('The R2 score for the validation set is: ' + str(R2_validation))

    # Calculates the root mean squared error for training, test and validation sets
    def __calculate_rmse__(self, y_predictions_train, y_predictions_test, y_predictions_validation):
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

def evaluate_all_models(model_dictionaries):
    model_comparisons = []
    for item in model_dictionaries:
        best_estimator, best_performance_metrics, best_hyperparameters = tune_regression_model_hyperparameters(model = item['model'], hyperparameters = item['hyperparameters'])
        model_comparisons.append({'estimator': best_estimator, 'metrics': best_performance_metrics, 'hyperparameters': best_hyperparameters})
        save_model(folder = item['folder'], model = best_estimator, metrics = best_performance_metrics, hyperparameters= best_hyperparameters)
    return model_comparisons

def find_best_model(dictionary):
    best_model = None
    hyperparams = None 
    performance_metrics = None
    r2 = -10^10
    for model in dictionary: 
        if model['metrics']['r2'] > r2:
            r2 = model['metrics']['r2']
            performance_metrics = model['metrics']
            hyperparams = model['hyperparameters']
            best_model = model['estimator']
    return best_model, hyperparams, performance_metrics

    
if __name__ == "__main__":
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data= 'cleaned_tabular_data.csv')
    #base_regression_model()

    SGDRegressor_hyperparameters = {
        'loss': ['epsilon_insensitive', 'squared_error', 'squared_epsilon_insensitive', 'huber'],
        'penalty': ['l2', 'l1', 'elasticnet'],
        'alpha': [0.00006, 0.00008, 0.0001, 0.00012, 0.00014, 0.00015,  0.00016, 0.00018],
        'max_iter': [3000]
        }

    #A greater number of different hyperparameters have been tested; this list is less expansive due to the time taken to compute all different combinations.
    DecisionTreeRegressor_hyperparameters = {
        'criterion': ['friedman_mse'],
        'splitter': ['best', 'random'],
        'max_depth': [4, 5, 6, 7],
        'min_samples_split': [3, 4, 5],
        'min_samples_leaf': [1, 2, 3],
        'min_weight_fraction_leaf': [0.0, 0.005, 0.01],
        'max_features': ['auto'],
        'random_state': [None, 1],
        'max_leaf_nodes': [None, 1],
        'min_impurity_decrease': [0.04, 0.05, 0.06],
        'ccp_alpha': [0.05, 0.075, 0.1, 0.125]
        }

    RandomForestRegressor_hyperparameters = {
        'n_estimators': [100],
        'criterion': ["poisson"],
        'min_samples_split': [2, 3, 4], 
        'min_samples_leaf': [1, 2, 3],
        'min_weight_fraction_leaf': [0.5, 0.01, 0.015], 
        'max_features': ['sqrt'],
        'min_impurity_decrease': [0.01],
        'verbose': [0],
        'ccp_alpha': [0.005, 0.01, 0.015] 
        }

    GradientBoostingRegressor_hyperparameters = {
        'loss': ['absolute_error'],
        'learning_rate': [0.07, 0.075, 0.08],
        'n_estimators': [100],
        'subsample': [1],
        'min_samples_leaf': [1],
        'max_depth': [None],
        'alpha': [0.75, 0.8, 0.85],
        'ccp_alpha': [0.0, 0.005]
        }

    model_dictionaries = [{'model': linear_model.SGDRegressor(), 'hyperparameters': SGDRegressor_hyperparameters, 'folder': 'Models/Regression/Linear_Regression/'}, 
    {'model': tree.DecisionTreeRegressor(), 'hyperparameters': DecisionTreeRegressor_hyperparameters, 'folder': 'Models/Regression/Decision_Tree_Regressor/'}, 
    {'model': ensemble.RandomForestRegressor(), 'hyperparameters': RandomForestRegressor_hyperparameters, 'folder': 'Models/Regression/Random_Forest_Regressor/'}, 
    {'model': ensemble.GradientBoostingRegressor(), 'hyperparameters': GradientBoostingRegressor_hyperparameters, 'folder': 'Models/Regression/Gradient_Boosting_Regressor/'}]

    model_comparisons = evaluate_all_models(model_dictionaries= model_dictionaries)
    best_estimator, best_performance_metrics, best_hyperparameters = find_best_model(dictionary= model_comparisons)
    print(best_estimator)
    print(best_hyperparameters)
    print(best_performance_metrics)

#'min_samples_split': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#        'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
 #       'max_features': ['auto', 'sqrt', 'log2'],
  #      'random_state': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
   #     'max_leaf_nodes': [None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #    'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
     #   'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]#


  #    'n_estimators': [100],
   #     'criterion': ["poisson"],
    #    'min_samples_split': [1, 2, 3], 3
     #   'min_samples_leaf': [1, 2], 2
      #  'min_weight_fraction_leaf': [0.0, 0.01], 0.01
       # 'max_features': ['sqrt'],
        #'min_impurity_decrease': [0.0, 0.01], 0.01
        #'verbose': [0],
        #'ccp_alpha': [0, 0.01] 0.01

# GradientBoostingRegressor_hyperparameters = {
#        'loss': ['squared_error', 'absolute_error', 'huber', 'quantile'],
 #       'learning_rate': [0.075, 0.1, 0.125],
  #      'n_estimators': [80, 100, 120],
   #     'subsample': [0.8, 0.9, 1],
    #    'min_samples_leaf': [0.5, 1, 2],
     #   'max_depth': [None, 3],
      #  'alpha': [0.8, 0.9, 1],
       # 'ccp_alpha': [0.0, 0.01, 0.02]
        #}

