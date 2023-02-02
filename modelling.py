import sklearn
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from sklearn import linear_model
from sklearn import metrics

X, y = load_airbnb('cleaned_tabular_data.csv')
#Splits the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Creates and trains the model
model = linear_model.SGDRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(metrics.mean_squared_error(y_pred, y_test))
