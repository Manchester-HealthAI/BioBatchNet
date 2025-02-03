import torch.nn as nn 
import torch

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, input_dim),
        )
    
    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return z, out

class DEC(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super(DEC, self).__init__()
        self.autoencoder = Autoencoder(input_dim, latent_dim=latent_dim)
        self.n_clusters = n_clusters
        self.latent_dim = latent_dim
        
        initial_cluster = torch.zeros(n_clusters, latent_dim, dtype=torch.float)
        self.clusters = nn.Parameter(initial_cluster)
    
    def forward(self, x):
        z, out = self.autoencoder(x)
        q = self.calculate_q(z)
        return q, z, out
    
    def calculate_q(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.clusters) ** 2, dim=2))
        q = q ** ((1 + 1) / 2.0)  
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
    
class ClusteringHead(nn.Module):
    def __init__(self, embedding_dim, num_clusters):
        super(ClusteringHead, self).__init__()
        self.clustering = nn.Sequential(
            nn.Linear(embedding_dim, 100),
            nn.ReLU(),
            nn.Linear(100, num_clusters),
            nn.Softmax()
        )

    def forward(self, x):
        return self.clustering(x)