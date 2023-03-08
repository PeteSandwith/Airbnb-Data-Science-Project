import torch
from torch.utils.data import Dataset
import pandas as pd




class AirbnbDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')
   
    def __getitem__(self, index):
        example = self.data.iloc[index]
        return example

    def __len__(self):
        return len(self.data)


data = AirbnbDataset()
print(data[10])
print(len(data))