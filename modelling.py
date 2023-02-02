import sklearn
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from sklearn import linear_model
from sklearn import metrics

X, y = load_airbnb('cleaned_tabular_data.csv')
# Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
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