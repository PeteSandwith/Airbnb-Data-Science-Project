import sklearn
from sklearn import metrics
import numpy as np
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from modelling import prepare_data
from modelling import save_model


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
    #f1_scorer = metrics.make_scorer(metrics.f1_score, average= "macro")
    grid = sklearn.model_selection.GridSearchCV(estimator= model, param_grid= hyperparameters, scoring= 'accuracy', verbose= 10)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    best_performance_metrics = {
        'f1_train': calculate_f1(X_train, y_train, best_estimator),
        'f1_test': calculate_f1(X_test, y_test, best_estimator),
        'f1_train': calculate_f1(X_train, y_train, best_estimator),
        'f1_test': calculate_f1(X_test, y_test, best_estimator),
        'precision_train': calculate_precision(X_train, y_train, best_estimator),
        'precision_test': calculate_precision(X_test, y_test, best_estimator),
        'recall_train': calculate_recall(X_train, y_train, best_estimator),
        'recall_test': calculate_recall(X_test, y_test, best_estimator),
        'accuracy_train': calculate_accuracy(X_train, y_train, best_estimator),
        'accuracy_test': calculate_accuracy(X_test, y_test, best_estimator),
        'accuracy_validation': calculate_accuracy(X_validation, y_validation, best_estimator)
    }
    best_hyperparameters = best_estimator.get_params()

    return best_estimator, best_performance_metrics, best_hyperparameters

def evaluate_all_models(model_dictionaries):
    model_comparisons = []
    for item in model_dictionaries:
        best_estimator, best_performance_metrics, best_hyperparameters = tune_classification_model_hyperparameters(model = item['model'], hyperparameters = item['hyperparameters'])
        model_comparisons.append({'estimator': best_estimator, 'metrics': best_performance_metrics, 'hyperparameters': best_hyperparameters})
        save_model(folder = item['folder'], model = best_estimator, metrics = best_performance_metrics, hyperparameters= best_hyperparameters)
    return model_comparisons

def find_best_model(dictionary):
    best_model = None
    hyperparams = None 
    performance_metrics = None
    accuracy = 0
    for model in dictionary: 
        if model['metrics']['accuracy_validation'] > accuracy:
            accuracy = model['metrics']['accuracy_validation']
            performance_metrics = model['metrics']
            hyperparams = model['hyperparameters']
            best_model = model['estimator']
    return best_model, hyperparams, performance_metrics



if __name__ == "__main__":

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_data(data = 'cleaned_tabular_data.csv', feature_columns= ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'], label_columns='Category')

    logistic_regression_hyperparameters = {
        'penalty': ['l2', None],
        'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
        'max_iter': [200],
        'C': [0.8, 0.9, 1.0, 1.1, 1.2],

        }

    decision_tree_classifier_hyperparameters = {
        'criterion': ['gini', 'entropy', 'log_loss'],
        'splitter': ['best', 'random'],
        'min_samples_split': [1, 2, 3],
        'min_samples_leaf': [0.5, 1, 1.5, 2],
        'ccp_alpha': [0, 0.005, 0.01, 0.015],
        'max_features': ['auto', 'sqrt', 'log2'],
        'min_weight_fraction_leaf': [0.0, 0.01, 0.02],
        'min_impurity_decrease': [0.0, 0.01, 0.02]
    }

    random_forest_classifier_hyperparameters = {
        'n_estimators': [90, 100, 110],
        'criterion': ['gini', 'entropy', 'log_loss'],
        'min_samples_split': [2, 3, 4],
        'min_weight_fraction_leaf': [0.0025, 0.05],
        'max_features': ['sqrt', 'log', None],
        'ccp_alpha': [0.0, 0.01]
    }

    gradient_boosting_classifier_hyperparameters = {
        'loss': ['log_loss', 'exponential'],
        'learning_rate': [0.05, 0.1, 0.15],
        'n_estimators': [90, 100, 110],
        'criterion': ['friedman_mse', 'squared_error'],
        'min_samples_leaf': [1, 2],
        'min_weight_fraction_leaf': [0.0, 0.1],
        'ccp_alpha': [0.0]
    }

    model_dictionaries = [{'model': linear_model.LogisticRegression(), 'hyperparameters': logistic_regression_hyperparameters, 'folder': 'Models/Classification/Logistic_Regression/'}, 
    {'model': tree.DecisionTreeClassifier(), 'hyperparameters': decision_tree_classifier_hyperparameters, 'folder': 'Models/Classification/Decision_Tree_Classifier/'}, 
    {'model': ensemble.RandomForestClassifier(), 'hyperparameters': random_forest_classifier_hyperparameters, 'folder': 'Models/Classification/Random_Forest_Classifier/'}, 
    {'model': ensemble.GradientBoostingClassifier(), 'hyperparameters': gradient_boosting_classifier_hyperparameters, 'folder': 'Models/Classification/Gradient_Boosting_Classifier/'}]



    model_comparisons = evaluate_all_models(model_dictionaries= model_dictionaries)
    best_model, hyperparams, performance_metrics = find_best_model(model_comparisons)
    print(best_model)
    print(hyperparams)
    print(performance_metrics)
    #save_model(model= best_estimator, metrics= best_performance_metrics, hyperparameters=best_hyperparameters, folder='Models/Classification/Logistic_Regression/')

    #model = sklearn.linear_model.LogisticRegression()
    #model.fit(X_train, y_train)
    #calculate_scores(X_train, X_test, y_train, y_test, model)

    