import torch.nn as nn
import torch.optim as optim
import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from utils import soft_assign, target_distribution, pairwise_loss, compute_coassociation_matrix, hierarchy_cluster_with_centroids

class Trainer:
    def __init__(self, model, cluster_optimizer, pretrain_optimizer, dec_dataloader, ml_dataloader, cl_dataloader, eva_dataloader, device):  
        self.model = model
        self.cluster_optimizer = cluster_optimizer
        self.pretrain_optimizer = pretrain_optimizer

        self.dec_dataloader = dec_dataloader
        self.ml_dataloader = ml_dataloader
        self.cl_dataloader = cl_dataloader
        self.eva_dataloader = eva_dataloader

        self.device = device
        self.criterion_recon = nn.MSELoss()  
        self.criterion_classification = nn.CrossEntropyLoss()
        self.alpha = 1
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.cluster_optimizer, mode='min', factor=0.1, patience=10)
 
    def pretrain_autoencoder(self, epochs, lr=1e-4):
        self.model.autoencoder.train()
        optimizer = optim.Adam(self.model.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_data in self.dec_dataloader:
                x, _ = batch_data
                x = x.to(self.device)
                optimizer.zero_grad()
                z, out = self.model.autoencoder(x)
                loss = criterion(out, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print("Pretraining finished")  
    
    def dec_epoch(self):
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        criterion_recon = nn.MSELoss()
        total_loss = 0.0
        total_kl_loss = 0.0
        total_recon_loss = 0.0

        for batch_data in self.dec_dataloader:
            x, _ = batch_data
            x = x.to(self.device)
            
            q, z, x_reconstructed = self.model(x)  
            p = target_distribution(q.detach())
            
            kl_loss = criterion_kl(torch.log(q), p)
            recon_loss = criterion_recon(x_reconstructed, x)
            loss = kl_loss + 0.1*recon_loss
            
            self.cluster_optimizer.zero_grad()
            loss.backward()
            self.cluster_optimizer.step()
            
            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_recon_loss += recon_loss.item()
    
    def constrain_epoch(self, epoch):
        update_ml = 1
        update_cl = 1

        ml_loss = 0.0
        total_samples = 0
        if epoch % update_ml == 0:
            all_iter = iter(self.dec_dataloader)

            for px1, px2 in self.ml_dataloader:
                try:
                    data_all, _ = next(all_iter)
                except StopIteration:
                    all_iter = iter(self.dec_dataloader)
                    data_all, _ = next(all_iter)

                px1, px2 = px1.to(self.device), px2.to(self.device)
                data_all = data_all.to(self.device)
                batch_size = px1.shape[0]
                total_samples += batch_size
                
                self.cluster_optimizer.zero_grad()

                q1, z1, xr1 = self.model(px1)
                q2, z2, xr2 = self.model(px2)

                _, _, xr_all = self.model(data_all)

                recon_loss_all = self.criterion_recon(data_all, xr_all)
                
                pair_recon = self.criterion_recon(px1, xr1) + self.criterion_recon(px2, xr2)
                
                loss = 1*pairwise_loss(q1, q2, "ML") + 0.2*pair_recon

                ml_loss += loss.item()
                loss.backward()
                self.cluster_optimizer.step()

        cl_loss = 0
        if epoch % update_cl == 0:
            all_iter = iter(self.dec_dataloader)

            for px1, px2 in self.cl_dataloader:
                try:
                    data_all, _ = next(all_iter)
                except StopIteration:
                    all_iter = iter(self.dec_dataloader)
                    data_all, _ = next(all_iter)

                px1, px2 = px1.to(self.device), px2.to(self.device)
                data_all = data_all.to(self.device)

                self.cluster_optimizer.zero_grad()
                q1, z1, xr1 = self.model(px1)
                q2, z2, xr2 = self.model(px2)   

                pair_recon = self.criterion_recon(px1, xr1) + self.criterion_recon(px2, xr2)        

                _, _, xr_all = self.model(data_all)
                recon_loss_all = self.criterion_recon(data_all, xr_all)

                loss = 1*pairwise_loss(q1, q2, "CL") + 0.2*pair_recon
                cl_loss += loss.item()
                loss.backward()

                self.cluster_optimizer.step()
    
    def train(self, pre_epochs=50, train_epochs=30, tolerance=0.005):
        self.pretrain_autoencoder(epochs=pre_epochs)
        cluster_centers = self.initialize_clusters()
        # self.initialize_clusters_eac()

        previous_predicted_labels = None

        for epoch in range(train_epochs):
            print(f"{epoch+1} pairwise train") 
            self.constrain_epoch(epoch)
            predicted_labels, true_labels, acc, ari, nmi = self.evaluate()

            print(f"{epoch+1}: ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}") 

            num_changed = np.sum(predicted_labels != previous_predicted_labels)
            change_ratio = num_changed / len(predicted_labels)

            print(f"Change ratio: {change_ratio*100:.2f}%")

            if change_ratio <= tolerance:
                print(f"Early stopping triggered. Change ratio {change_ratio*100:.2f}% <= tolerance {tolerance*100}%")
                break

            previous_predicted_labels = predicted_labels.copy()
            
        # cluster_centers = self.initialize_clusters()
        # for epoch in range(train_epochs):
        #     print(f"开始第 {epoch+1} 个聚类周期")  # Starting epoch
        #     self.dec_epoch()
        #     self.constrain_epoch(epoch)
        #     _, _, acc, ari, nmi = self.evaluate()
        #     print(f"{epoch+1}: 准确率={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")  # Evaluation results
            
        #     predicted_labels, true_labels, acc, ari, nmi = self.evaluate()
        #     num_changed = np.sum(predicted_labels != previous_predicted_labels)
        #     change_ratio = num_changed / len(predicted_labels)

        #     print(f"Change ratio: {change_ratio*100:.2f}%")

        #     if change_ratio <= tolerance:
        #         print(f"Early stopping triggered. Change ratio {change_ratio*100:.2f}% <= tolerance {tolerance*100}%")
        #         break

        #     previous_predicted_labels = predicted_labels.copy()
        print("Training Finished")  # Training completed

    def initialize_clusters(self):
        print("Initializing cluster centers with K-Means")  
        self.model.autoencoder.eval()
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch_data in self.dec_dataloader:
                x, y = batch_data
                x = x.to(self.device)
                z, _ = self.model.autoencoder(x)
                embeddings.append(z.detach().cpu().numpy())
                labels.append(y.detach().cpu().numpy())
        embeddings = np.vstack(embeddings)
        labels = np.concatenate(labels, axis=0) 
        
        kmeans = KMeans(n_clusters=self.model.n_clusters, init='k-means++', n_init=20, random_state=42)
        y_pred = kmeans.fit_predict(embeddings)
        
        self.model.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

        return kmeans.cluster_centers_
    
    def initialize_clusters_eac(self):
        centroids = self.centroid_by_eac()
        self.model.clusters.data = torch.tensor(centroids).to(self.device)

    def evaluate(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_data in self.eva_dataloader:
                x, y = batch_data
                x = x.to(self.device)
                q, _, _ = self.model(x)
                all_preds.append(q.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        preds = np.vstack(all_preds)
        true_labels = np.concatenate(all_labels, axis=0)
        
        # Predict cluster assignments
        predicted_labels = np.argmax(preds, axis=1)
        
        # Compute metrics
        acc = self.cluster_accuracy(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
          
        return predicted_labels, true_labels, acc, ari, nmi
    
    def cluster_accuracy(self, y_true, y_pred):
        from scipy.optimize import linear_sum_assignment
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] +=1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i,j] for i,j in zip(ind[0], ind[1])]) / y_pred.size

    def centroid_by_eac(self):
        self.model.autoencoder.eval()  
        all_z = self.load_data_from_dataloader()

        # calculate cluster center
        num_samples = int(all_z.size(0) * 0.1)
        global_sample_indices = torch.randperm(all_z.size(0), device=self.device)[:num_samples]
        samples = all_z[global_sample_indices]

        # build co-matrix using batch effect weights 
        # weights = compute_weight(samples, samples_batch_id)
        co_matrix, _ = compute_coassociation_matrix(samples, n_clusters=7)

        # select high confidence points
        clusters, centroids = hierarchy_cluster_with_centroids(samples, co_matrix, n_clusters=7)

        return centroids
    
    def load_data_from_dataloader(self):

        self.model.autoencoder.eval()
        all_z = []

        with torch.no_grad():
            for data, _ in self.dec_dataloader:
                data = data.to(self.device)
                z, out = self.model.autoencoder(data)
                all_z.append(z)

        all_z = torch.cat(all_z, dim=0)
        return all_z

    def get_embedding(self):
        self.model.eval()

        all_embedding = []
        all_label = []
        with torch.no_grad():
            for data, label in self.dec_dataloader:
                data = data.to(self.device)
                z, _ = self.model.autoencoder(data)
                all_embedding.append(z)
                all_label.append(label)

        embeddings = torch.cat(all_embedding, dim=0).cpu().numpy()
        lables = np.concatenate(all_label, axis=0)
        return embeddings, lables#



