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
- A grid search was performed over a suitable range of hyperparameter values, in order to tune the hyperparameters of the model. This was done both using a custom gridsearch function which was implemented from scratch, and then by using the inbuilt sklearn GridSearchCV function. 
```
def tune_regression_model_hyperparameters(model, hyperparameters):
    grid = sklearn.model_selection.GridSearchCV(estimator= model, param_grid= hyperparameters, scoring= 'r2', refit= 'r2', verbose= 10)
    grid.fit(X_train, y_train)
    
    best_estimator = grid.best_estimator_
    best_performance_metrics = {'r2': calculate_validation_r2(model = best_estimator), 'rmse': calculate_validation_rmse(model= best_estimator)}
    best_hyperparameters = best_estimator.get_params()

    return best_estimator, best_performance_metrics, best_hyperparameters
```
- A function was created to save the trained model, its tuned hyperparameters and its performance metrics. Whilst the hyperparameters and metrics were simply saved in json files, the model itself was saved using the python library joblib. Joblib can be used to easily save and load machine learning models. 

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
- Several other models provided by sklearn were implemented in addition to the SGD regressor: decision trees, random forests, and gradient boosting. All of these models were tuned to find optimum hyperparameters. The following function takes in a dictionary containing each different model type along with the range of hyperparameters to be searched over. It returns a dictionary containing, for each model, the name of the model, the best hyperparameters and the performance metrics.
```
def evaluate_all_models(model_dictionaries):
    model_comparisons = []
    for item in model_dictionaries:
        best_estimator, best_performance_metrics, best_hyperparameters = tune_regression_model_hyperparameters(model = item['model'], hyperparameters = item['hyperparameters'])
        model_comparisons.append({'estimator': best_estimator, 'metrics': best_performance_metrics, 'hyperparameters': best_hyperparameters})
        save_model(folder = item['folder'], model = best_estimator, metrics = best_performance_metrics, hyperparameters= best_hyperparameters)
    return model_comparisons
```
## Milestone 4
- This process was repeated, this time for a classification problem. The same 'prepare_data' function was imported and used to load in the airbnb dataset, using the 'category' as the label. Several sklearn classification models were trained and tuned: logistic regression, decision tree classifier, random forest classifier, gradient boosting classifier. Many of the functions used in the regression problem could be repurposed to analyse the classifier models, for instance the function used to determine the best model:
```
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
```
## Milestone 5
- A configurable neural network was created to solve the regression problem. A dataloader was created to shuffle and batch the data to be fed into the neural network:
```
def create_dataloaders(dataset, batch_size):
    # Splits dataset into train, test and validation sets
    train_set, test_set = random_split(dataset, [0.5, 0.5])
    test_set, validation_set = random_split(test_set, [0.5, 0.5])

    # Creates and returns dataloaders for the train, test and validation sets
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_set, batch_size=batch_size, shuffle=True)

    dataloader_dict = {'Train': train_loader, 'Test': test_loader, 'Validation': validation_loader}
    return dataloader_dict
```
- A function 'train' was created which takes in the model, the dataloader and the number of epochs. This function performs forward passes on the batches of data and outputs a prediction. It iteratively optimises the parameters of the model, whilst calculating the mse for each batch. Later on, the function was adapted to take in a dictionary 'config' which specifies the hyperparameters of the model. It also uses the SummaryWriter object to visualise the mse metric using tensorflow.
```
def train(model, dataloader, config, number_epochs=10):

    # Defines the optimiser to be used, in this case stochastic gradient descent
    optimiser = convert_optimiser_to_callable(config['Optimiser'])(model.parameters(), lr= config["Learning_rate"])
    
    train_mse_loss = []
    validation_mse_loss = []
    # Initialises SummaryWriter
    writer = SummaryWriter()
    # Variable to track the overall batch number
    batch_index_train = 0
    batch_index_validation = 0
    for epoch in range(number_epochs):
        for batch in dataloader['Train']:
                features, labels = batch
                predictions = model(features).squeeze()
                mse_loss = Functional.mse_loss(predictions, labels)
                #Populates the grad attribute of the parameters 
                mse_loss.backward()
                #Optimisation step: optimises parameters based on their grad attribute 
                optimiser.step()
                # Resets the grad attributes of the parameters, which are otherwise stored
                optimiser.zero_grad()
                print(mse_loss.item())
                writer.add_scalar('mse_loss_train', mse_loss.item(), batch_index_train)
                train_mse_loss.append(mse_loss.item())
                batch_index_train += 1
```
- In order to tune the hyperparameters of the model, such as the type of optimiser used and the width / depth of the hidden layers in the neural network, a function called find_best_nn was defined. It uses the generate_nn_configs function to generate a list of dictionaries, each one of which contains a particular set of hyperparameters in the grid search. Find_best_nn() then trains a neural network using each different set of hyperparameters and compares the performance metrics to determine the best model.
```
def find_best_nn():
    configuration_list = generate_nn_configs()
    best_mse = 10000000
    best_configs = {}
    best_model = 'Empty'
    best_metrics = {}
    for config in configuration_list:
        model = PyTorchModel(number_inputs=11, number_outputs=1, config=config)
        metrics = train(model=model, dataloader=dataloader_dict, config=config)
        if metrics["MSE Validation"] <= best_mse:
            best_mse = metrics["MSE Validation"]
            best_metrics = metrics
            best_configs = config
            best_model = model
        
    return best_model, best_configs, best_metrics
```
- The best model, along with its hyperparameters and metrics, can be saved using the function below. The function uses a combination of the os and the datetime python libraries to create unique directories within which to save a copy of the model, and json files containing its hyperparameters and performance metrics. The function also checks to ensure that the object being saved is indeed a PyTorch model.
```
def save_model(folder, model, config, metrics):
    if type(model) == PyTorchModel:
        date = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        current_directory = os.getcwd()
        folder = folder + date 
        if os.path.exists(folder) == False:
            os.mkdir(folder)
        
        model_filename = folder + '/model.joblib'
        hyperparameters_filename = folder + '/hyperparameters.json'
        performance_metrics_filename = folder + '/metrics.json'

        # Saves the model 
        joblib.dump(model, os.path.join(current_directory, model_filename))

        # Saves the hyperparameters
        with open(os.path.join(current_directory, hyperparameters_filename), "w") as file:
            json.dump(config, file) 

        # Saves the metrics
        with open(os.path.join(current_directory, performance_metrics_filename), "w") as file:
            json.dump(metrics, file) 

    else:
        return "This object is not a PyTorch model"
```