import torch
from torch.utils.data import Dataset
import pandas as pd
from tabular_data import load_airbnb



class AirbnbDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.X, self.y = load_airbnb(file= 'cleaned_tabular_data.csv',
        feature_columns = ['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms'],
        label_columns = 'Price_Night')
   
    def __getitem__(self, idx):
        return (torch.tensor(self.X.iloc[idx]), torch.tensor(self.y.iloc[idx]))

    def __len__(self):
        return len(self.X)


data = AirbnbDataset()
print(data[11])
print(type(data))