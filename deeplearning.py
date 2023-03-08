import torch
from torch.utils.data import Dataset
import pandas as pd

data = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')


class AirbnbDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')
   
    def __getitem__(self, index):
        example = self.data.iloc[index]
        features = example['guests', 'beds', 'bathrooms', 'Cleanliness_rating', 'Accuracy_rating', 'Communication_rating', 'Location_rating', 'Check-in_rating', 'Value_rating', 'amenities_count', 'bedrooms']
        label = example['Price_Night']
        return features, label