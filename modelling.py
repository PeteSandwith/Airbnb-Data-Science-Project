import sklearn
from sklearn.model_selection import train_test_split
from tabular_data import load_airbnb

X, y = load_airbnb('cleaned_tabular_data.csv')

