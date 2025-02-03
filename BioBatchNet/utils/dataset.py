from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
from pathlib import Path

class GeneralDataset(Dataset):
    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_configs = {
        'Damond': {
            'feature_cols': (0, 38),
            'batch_col': 40,
            'cell_type_col': 38,
            'datadir': BASE_DIR / 'Data/IMC_data/csv_format/Damond_2019_Pancreas.csv'
        },

        'Hoch': {
            'feature_cols': (0, 41),
            'batch_col': 42,
            'cell_type_col': 41,
            'datadir': BASE_DIR / 'Data/IMC_data/csv_format/HochSchulz.csv'
        },

        'IMMU': {
            'feature_cols': (0, 40),
            'batch_col': 41,
            'cell_type_col': 40,
            'datadir': BASE_DIR / 'Data/IMC_data/csv_format/IMMUcan_batch.csv'
        },


    }

    def __init__(self, dataset_name):
        super().__init__()

        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Dataset {dataset_name} is not recognized")
        
        config = self.dataset_configs[dataset_name]
        feature_cols = config['feature_cols']
        cell_cols = config['cell_type_col']
        batch_col = config['batch_col']
        data_dir = config['datadir']

        self.data = pd.read_csv(data_dir)
        self.features = torch.tensor(self.data.iloc[:, feature_cols[0]:feature_cols[1]].values.astype(np.float32))
        
        # Retain cell type as both codes and categories
        cell_type_categorical = pd.Categorical(self.data.iloc[:, cell_cols])
        self.cell_type = torch.tensor(cell_type_categorical.codes, dtype=torch.int64)
        self.cell_type_names = cell_type_categorical.categories  # Store the original names
        
        self.batch_id = torch.tensor(pd.Categorical(self.data.iloc[:, batch_col]).codes, dtype=torch.int64)

    def __getitem__(self, index):
        return self.features[index], self.batch_id[index], self.cell_type[index]

    def __len__(self):
        return len(self.data)

class MLDataset(Dataset):
    def __init__(self, ml_ind1, ml_ind2, data):
        self.ml_pairs = list(zip(ml_ind1, ml_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.ml_pairs)
    
    def __getitem__(self, idx):
        i, j = self.ml_pairs[idx]
        sample_i = self.data[i]
        sample_j = self.data[j]
        return sample_i, sample_j

class CLDataset(Dataset):
    def __init__(self, cl_ind1, cl_ind2, data):
        self.cl_pairs = list(zip(cl_ind1, cl_ind2))
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.cl_pairs)
    
    def __getitem__(self, idx):
        i, j = self.cl_pairs[idx]
        sample_i = self.data[i]
        sample_j = self.data[j]
        return sample_i, sample_j

class GeneDataset(Dataset):
    def __init__(self, data_dir):
        super().__init__()

        self.adata = pd.read_csv(data_dir)
        self.data = self.adata.iloc[:, 0:2000].values
        self.cell_type = pd.Categorical(self.adata['cell_type']).codes
        self.batch = pd.Categorical(self.adata['batch']).codes

    def __len__(self):
        return self.adata.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float32)
        cell_type = torch.tensor(self.cell_type[index], dtype=torch.long)
        batch = torch.tensor(self.batch[index], dtype=torch.long)
        return data, batch, cell_type


    