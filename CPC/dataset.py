import torch
from torch.utils.data import Dataset

class GeneralDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = torch.tensor(data, dtype=torch.float32)  
        self.label = torch.tensor(label, dtype=torch.long)   
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
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
