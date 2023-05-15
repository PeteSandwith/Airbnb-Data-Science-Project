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
#print(data[11])
#print(type(data))


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

#config = get_config_dict('configurations.yaml')
#print(config)
config = {"Optimiser": 'SGD', "Learning_rate": 0.0001, "Hidden_layer_width": 20, "Hidden_layer_depth": 2}


def convert_optimiser_to_callable(string):
    '''Function that converts the description of the optimiser from the configuration dictionary, which is a string, to a callable object that can be used by pytorch'''

    if string == 'SGD':
        optimiser = torch.optim.SGD
    return optimiser


def train(model, dataloader, config, number_epochs=10):

    # Defines the optimiser to be used, in this case stochastic gradient descent
    optimiser = config['Optimiser'](model.parameters(), lr= config["Learning_rate"])

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

                batch_index_train += 1


        for batch in dataloader['Validation']:
            features, labels = batch
            predictions = model(features).squeeze()
            mse_loss = Functional.mse_loss(predictions, labels)
            print(mse_loss.item())
            writer.add_scalar('mse_loss_validation', mse_loss.item(), batch_index_validation)
            batch_index_validation += 1

if __name__ == '__main__':
    pass

    model = PyTorchModel(number_inputs=11, number_outputs=1, config=config)
    train(model=model, dataloader=dataloader_dict, config=config)
    
    #y_hat = model(data[0][0])
    #print("Weight:", model.linear_layer.weight)
    #print("Bias:", model.linear_layer.bias)
    #print("Predictions:", y_hat)