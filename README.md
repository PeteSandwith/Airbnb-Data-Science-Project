# Airbnb-Data-Science-Project
An industry grade data science project. A pipeline was created to systematically clean and perform feature engineering on airbnb data samples. Multiple machine learning models were trained; hyperparameters of the models were tuned to improve performance and iterative selection was used to add features to enhance the accuracy of the models. Tools used to deploy this system include AWS, PyTorch, Pandas, SkLearn and Tensorboard.

## Milestone 1
- A python script tabular_data.py was written to systematically load and clean the data relating to airbnb property listings. To manipulate the data set we chose to use the python library pandas, which is an extremely popular tool for data analysis and manipulation. By loading the dataset as a pandas dataframe actions can be performed on the whole dataset, based on certain conditions. For example, any items that had a missing value in the column relating to the number of beds were given a value of 1:
```
dataframe['beds'].where(~dataframe['beds'].isna(), 1, inplace = True)
```
whilst the presence of missing data in some columns meant that the entire row of the table had to be removed:
```
dataframe.dropna(axis = 0, how = 'any', subset = ['Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating'], inplace = True)
```
- The description column contained text data that was saved in the form of a list of strings; however due to the presence of quotations pandas didn't recognise the objects as lists but rather string objects whose contents were valid string. A specific function was created to parse this data using the ast package so that it could be saved as a single string object with no erroneous whitespace:
```
def parses_description_strings(string):
    try:
        description_as_list = ast.literal_eval(string)
        description_as_list.pop(0)
        
        number_empty_quotes = description_as_list.count('')
        for index in range(0,number_empty_quotes):
            description_as_list.remove('')
        
        cleaned_description = ' '.join(description_as_list)
        return cleaned_description
    except: 
        return string
```
- Finally all of the code used to clean the data was packaged into a single function, clean_tabular_data, and the cleaned data set was saved in a separate csv file, ready to be used to train ML models. 

## Milestone 2
- A second python script, prepare_image_data.py, was created in order to format and prepare the image data relating to each airbnb listing. Each image was loaded into the file, processed using the OpenCV library and then saved in a programatically determined location in the project folder. The processing of the images included resizing each image so that all images were the same height whilst retaining their original aspect ratio:
```
def resize_image(image):
    dimensions = image.shape
    scale = 400 / dimensions[0]
    image = cv.resize(image, (0,0), fx = scale, fy = scale)
    return image
```
## Milestone 3
- With the data preparation complete it was time to begin training machine learning models on the data, starting with regression models. A function load_airbnb was created that loaded the cleaned data in the form of a tuple (X,y):
```
def load_airbnb(file):
    dataset2 = pd.read_csv('airbnb-property-listings/tabular_data/{}'.format(file), index_col= 0)
    features = dataset2[['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']]
    labels = dataset2['Price_Night']
    return (features,labels)
```
where X was a pandas dataframe containing the features, or the input data used by the models to generate predictions, and y was pandas series containing the labels, or the data that we want the models to predict. In this case the features data contained a number of different variables for each property, and the label was the price per night of the property. 
- The first model to be implemented was the SGDRegressor model. This is a built-in model class from scikit-learn, a python library that contains various machine learning tools and algorithms. SGD stands for stochastic gradient descent, a reference to the iterative process used by the algorithm to minimise the loss function of the model. In order to improve the model's performance the features data were first normalised:
```
X = sklearn.preprocessing.normalize(X, norm='l2')
```
The features and label data were then split into training, test and validation sets:
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state= 2)
X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state= 2)
```
- It is important to split the data in this way so that when the time comes to comparing different models and tuning hyperparameters, the comparisons are not being made on data that the model had access to during training. For example, if a model has a high capacity it will be more prone to overfitting. An overfit model will of course appear to perform very well when evaluated on the training data set, but may perform poorly on previously unseen data. In this instance, comparing the overfit model to other models using the validation data set will be far more useful because it gives a better measure of how well the model will generalise to new data.
- The performance of models can be evaluated using a wide variety of different metrics provided by scikit-learn. The two metrics chosen to evaluate the SGDRegressor model were the R2 Score and the root mean squared error:
```
def __calculate_R2__(self, y_predictions_train, y_predictions_test, y_predictions_validation):
        R2_train = metrics.r2_score(y_train, y_predictions_train)
        R2_test = metrics.r2_score(y_test, y_predictions_test)
        R2_validation = metrics.r2_score(y_validation, y_predictions_validation)
        print('The R2 score for the training set is: ' + str(R2_train))
        print('The R2 score for the test set is: ' + str(R2_test))
        print('The R2 score for the validation set is: ' + str(R2_validation))
```
- A function was created to save the trained model, its hyperparameters and its performance metrics. Whilst the hyperparameters and metrics were simply saved in json files, the model itself was saved using the python library joblib. Joblib can be used to easily save and load machine learning models. 

```
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
```