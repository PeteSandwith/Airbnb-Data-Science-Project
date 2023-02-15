import sklearn
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from sklearn import linear_model
from sklearn import metrics
import numpy as np
from sklearn.preprocessing import normalize

np.random.seed(2)
X, y = load_airbnb('cleaned_tabular_data.csv')
# Normalises the feature data
X = sklearn.preprocessing.normalize(X, norm='l2')
# Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state= 2)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state= 2)
# Creates and trains the model
model = linear_model.SGDRegressor()
model.fit(X_train, y_train)

# Generates label predictions for the training and test sets
y_predictions_train = model.predict(X_train)
y_predictions_test = model.predict(X_test)

# Calculates the R^2 regression score function for training and test sets
R2_train = metrics.r2_score(y_train, y_predictions_train)
R2_test = metrics.r2_score(y_test, y_predictions_test)
print('The R2 score for the training set is: ' + str(R2_train))
print('The R2 score for the test set is: ' + str(R2_test))

# Calculates the root mean squared error for training and test sets
rmse_train = metrics.mean_squared_error(y_train, y_predictions_train, squared = False)
rmse_test = metrics.mean_squared_error(y_test, y_predictions_test, squared = False)
print('The rmse for the training set is: ' + str(rmse_train))
print('The rmse for the test set is: ' + str(rmse_test))


def custom_tune_regression_model_hyperparameters(dataset, hyperparameters):
    hyperparameter_combinations = []
    best_model = {'model': 0, 'validation_RMSE': 10000}
    for loss in hyperparameters['loss']:
        for penalty in hyperparameters['penalty']:
            for alpha in hyperparameters['alpha']:
                hyperparameter_combinations.append({'loss': loss, 'penalty': penalty, 'alpha': alpha})

    for combination in hyperparameter_combinations:
        model = linear_model.SGDRegressor(loss = combination['loss'], penalty = combination['penalty'], alpha = combination['alpha'], max_iter= 3000)
        model.fit(dataset['X_train'], dataset['y_train'])
        y_predictions_validation = model.predict(dataset['X_validation'])
        rmse = metrics.mean_squared_error(dataset['y_validation'], y_predictions_validation, squared = False)
        if rmse < best_model['validation_RMSE']:
            best_model['model'] = model
            best_model['validation_RMSE'] = rmse

    return best_model

hyperparameters = {'loss': ['epsilon_insensitive', 'squared_error', 'squared_epsilon_insensitive', 'huber'], 'penalty': ['l2', 'l1', 'elasticnet'], 'alpha': [0.0006, 0.0008, 0.001, 0.0012, 0.0014, 0.0015,  0.0016, 0.0018]}
custom_tune_regression_model_hyperparameters(dataset={'X_train': X_train, 'y_train': y_train, 'X_validation': X_validation, 'y_validation': y_validation}, hyperparameters= hyperparameters)