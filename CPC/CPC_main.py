import argparse
import collections
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from utils import generate_random_pair, transitive_closure
from CPC.CPC_trainer import Trainer
from model import DEC
from dataset import GeneralDataset, MLDataset, CLDataset

SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main():
    
    # prepare data
    table = pd.read_csv('data/IMMU_embedding.csv')
    table = table[table['cell type'] != 'undefined']

    data = table.iloc[:, 0:20].values
    cell_type = table.iloc[:, 21].values
    
    label_encoder = LabelEncoder()
    cell_type = label_encoder.fit_transform(cell_type)

    num_constraints = 4000
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(cell_type, num_constraints*2)
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, data.shape[0])

    ml_ind1 = ml_ind1[:num_constraints]
    ml_ind2 = ml_ind2[:num_constraints]
    cl_ind1 = cl_ind1[:num_constraints]
    cl_ind2 = cl_ind2[:num_constraints]

    train_dataset = GeneralDataset(data, cell_type)
    ml_dataset = MLDataset(ml_ind1, ml_ind2, data)
    cl_dataset = CLDataset(cl_ind1, cl_ind2, data)

    dec_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=256)
    ml_dataloader = DataLoader(ml_dataset, shuffle=True, batch_size=128)
    cl_dataloader = DataLoader(cl_dataset, shuffle=True, batch_size=128)

    # build model
    idea_model = DEC(input_dim=20, latent_dim=10, n_clusters=7)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    idea_model = idea_model.to(device)

    trainable_params = list(filter(lambda p: p.requires_grad, idea_model.autoencoder.parameters())) 
    pretrain_optimizer = torch.optim.Adam(trainable_params, lr=1e-3)
    cluster_optimizer = torch.optim.Adam(idea_model.parameters(), lr=1e-3)

    trainer = Trainer(idea_model, 
                      cluster_optimizer, 
                      pretrain_optimizer, 
                      dec_dataloader, 
                      ml_dataloader, 
                      cl_dataloader, 
                      device)
    
    trainer.train()


if __name__ == '__main__':
    main()
    