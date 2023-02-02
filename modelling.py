import sklearn
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb
from sklearn import linear_model

X, y = load_airbnb('cleaned_tabular_data.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = linear_model.SGDRegressor()
