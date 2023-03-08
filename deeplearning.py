import torch
from torch.utils.data import Dataset
import pandas as pd

data = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')


class AirbnbDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv('airbnb-property-listings/tabular_data/cleaned_tabular_data.csv')
   
