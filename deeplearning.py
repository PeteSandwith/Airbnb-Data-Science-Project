from cgi import test
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch.nn.functional as Functional
import pandas as pd
from tabular_data import load_airbnb
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

    return train_loader, test_loader, validation_loader

train_loader, test_loader, validation_loader = create_dataloaders(dataset= data, batch_size=12)
dataloader_dict = {'Train': train_loader, 'Test': test_loader, 'Validation': validation_loader}
class PyTorchModel(torch.nn.Module):

    # Constructor 
    def __init__(self, number_inputs, number_outputs):
        super().__init__()
        self.linear_layer = torch.nn.Linear(number_inputs, 12)
        self.linear_layer2 = torch.nn.Linear(12, number_outputs)
    
    # Forward method that will be run whenever we call the model
    def forward(self, X):
        X = self.linear_layer(X)
        X = Functional.relu(X)
        return self.linear_layer2(X)

def train(model, dataloader, number_epochs=10):

    # Defines the optimiser to be used, in this case stochastic gradient descent
    optimiser = torch.optim.SGD(model.parameters(), lr= 0.005)

    # Initialises SummaryWriter
    writer = SummaryWriter()
    # Variable to track the overall batch number
    batch_index = 0
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
                #print(mse_loss.item())
                writer.add_scalar('mse_loss', mse_loss.item(), batch_index)

                batch_index += 1


        for batch in dataloader['Validation']:
            features, labels = batch
            predictions = model(features).squeeze()
            mse_loss = Functional.mse_loss(predictions, labels)
            print(mse_loss.item())

if __name__ == '__main__':
    model = PyTorchModel(number_inputs=11, number_outputs=1)
    train(model=model, dataloader=dataloader_dict)
    
    #y_hat = model(data[0][0])
    #print("Weight:", model.linear_layer.weight)
    #print("Bias:", model.linear_layer.bias)
    #print("Predictions:", y_hat)