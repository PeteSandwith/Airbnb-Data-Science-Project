import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pandas as pd
from tabular_data import load_airbnb



class AirbnbDataset(Dataset):
    def __init__(self, file):
        super().__init__()
        self.X, self.y = load_airbnb(file= file,
        feature_columns = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'],
        label_columns = 'Price_Night')
   
    def __getitem__(self, idx):
        return (torch.tensor(self.X.iloc[idx]), torch.tensor(self.y.iloc[idx]))

    def __len__(self):
        return len(self.X)


data = AirbnbDataset(file = 'cleaned_tabular_data.csv')
#print(data[11])
#print(type(data))

train_set, test_set = random_split(data, [0.5, 0.5])
test_set, validation_set = random_split(test_set, [0.5, 0.5])
train_loader = DataLoader(dataset=train_set, batch_size=12, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=12, shuffle=True)
validation_loader = DataLoader(dataset=validation_set, batch_size=12, shuffle=True)


