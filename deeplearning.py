from cgi import test
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as Functional
import pandas as pd
from tabular_data import load_airbnb
import yaml
from torch.utils.tensorboard import SummaryWriter
import os
import joblib
import json
import datetime

class AirbnbDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.X, self.y = load_airbnb(file= file,
        feature_columns = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'],
        label_columns = 'Price_Night')
   
    def __getitem__(self, idx):
        return (torch.tensor(self.X.iloc[idx]).float(), torch.tensor(self.y.iloc[idx]).float())

    def __len__(self):
        return len(self.X)


data = AirbnbDataset(file = 'cleaned_tabular_data.csv')



#Function which takes in the dataset and creates train, test and validation dataloaders

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

dataloader_dict = create_dataloaders(dataset= data, batch_size=12)

class PyTorchModel(torch.nn.Module):

    # Constructor 
    def __init__(self, number_inputs, number_outputs, config):
        super().__init__()
        layers = []
        input_layer = torch.nn.Linear(number_inputs, config["Hidden_layer_width"])
        output_layer = torch.nn.Linear(config["Hidden_layer_width"], number_outputs)
        layers.append(input_layer)
        layers.append(torch.nn.ReLU())

        #Creates the hidden layers
        for layer in range(config['Hidden_layer_depth']):
            layers.append(torch.nn.Linear(config["Hidden_layer_width"], config["Hidden_layer_width"]))
            layers.append(torch.nn.ReLU())
        
        layers.append(output_layer)
        self.layers = torch.nn.Sequential(*layers)
        
    
    # Forward method that will be run whenever we call the model
    def forward(self, X):
       return self.layers(X)

def get_config_dict(file):
    with open(file, 'r') as f:
        config = yaml.safe_load(f)
        return config

config = get_config_dict('configurations.yaml')
print(config)


def convert_optimiser_to_callable(string):
    '''Function that converts the description of the optimiser from the configuration dictionary, which is a string, to a callable object that can be used by pytorch'''

    if string == 'SGD':
        optimiser = torch.optim.SGD
    elif string == 'Adagrad':
        optimiser = torch.optim.Adagrad
    elif string == 'Adam':
        optimiser = torch.optim.Adam
    return optimiser



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


        for batch in dataloader['Validation']:
            features, labels = batch
            predictions = model(features).squeeze()
            mse_loss = Functional.mse_loss(predictions, labels)
            print(mse_loss.item())
            writer.add_scalar('mse_loss_validation', mse_loss.item(), batch_index_validation)
            batch_index_validation += 1
            validation_mse_loss.append(mse_loss.item())
    
    mse_train = sum(train_mse_loss) / len(train_mse_loss)
    mse_validation = sum(validation_mse_loss) / len(validation_mse_loss)
    performance_metrics = {"MSE Training": mse_train, "MSE Validation": mse_validation}
    return performance_metrics

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

def generate_nn_configs():
    configs = []
    optimisers = ['SGD', 'Adagrad', 'Adam']
    learning_rates = [0.0005, 0.001, 0.0015]
    widths = [5, 7, 9]
    depths = [5, 7, 9]
    for optimiser in optimisers:
        for rate in learning_rates:
            for width in widths:
                for depth in depths:
                    configs.append({'Optimiser': optimiser, 'Learning_rate': rate, 'Hidden_layer_width': width, 'Hidden_layer_depth': depth})
    print(configs)

if __name__ == '__main__':
    pass
    generate_nn_configs()
    #model = PyTorchModel(number_inputs=11, number_outputs=1, config=config)
    #metrics = train(model=model, dataloader=dataloader_dict, config=config)
    #save_model("neural_networks/regression/", model, config, metrics)
    
  