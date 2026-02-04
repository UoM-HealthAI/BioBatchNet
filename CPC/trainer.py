import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from utils import pairwise_loss


class Trainer:
    def __init__(self, model, optimizer, train_dataloader, ml_dataloader, cl_dataloader, device):
        self.model = model
        self.optimizer = optimizer

        self.train_dataloader = train_dataloader
        self.ml_dataloader = ml_dataloader
        self.cl_dataloader = cl_dataloader

        self.device = device
        self.criterion_recon = nn.MSELoss()

    def pretrain(self, epochs, lr=1e-4):
        """Pretrain autoencoder with reconstruction loss."""
        self.model.autoencoder.train()
        optimizer = optim.Adam(self.model.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            total_loss = 0
            for x, _ in self.train_dataloader:
                x = x.to(self.device)
                optimizer.zero_grad()
                z, out = self.model.autoencoder(x)
                loss = criterion(out, x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        print("Pretraining finished")

    def initialize_clusters(self):
        """Initialize cluster centers with KMeans."""
        print("Initializing cluster centers with K-Means")
        self.model.autoencoder.eval()
        embeddings = []

        with torch.no_grad():
            for x, _ in self.train_dataloader:
                x = x.to(self.device)
                z, _ = self.model.autoencoder(x)
                embeddings.append(z.cpu().numpy())

        embeddings = np.vstack(embeddings)

        kmeans = KMeans(n_clusters=self.model.n_clusters, init='k-means++', n_init=20, random_state=42)
        kmeans.fit(embeddings)

        self.model.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(self.device)

    def train_epoch(self):
        """Train one epoch with pairwise constraints."""
        self.model.train()

        # Must-link training
        for x1, x2 in self.ml_dataloader:
            x1, x2 = x1.to(self.device), x2.to(self.device)

            self.optimizer.zero_grad()

            q1, z1, xr1 = self.model(x1)
            q2, z2, xr2 = self.model(x2)

            recon_loss = self.criterion_recon(x1, xr1) + self.criterion_recon(x2, xr2)
            ml_loss = pairwise_loss(q1, q2, "ML")
            loss = ml_loss + 0.1 * recon_loss

            loss.backward()
            self.optimizer.step()

        # Cannot-link training
        for x1, x2 in self.cl_dataloader:
            x1, x2 = x1.to(self.device), x2.to(self.device)

            self.optimizer.zero_grad()

            q1, z1, xr1 = self.model(x1)
            q2, z2, xr2 = self.model(x2)

            recon_loss = self.criterion_recon(x1, xr1) + self.criterion_recon(x2, xr2)
            cl_loss = pairwise_loss(q1, q2, "CL")
            loss = cl_loss + 0.1 * recon_loss

            loss.backward()
            self.optimizer.step()

    def train(self, pre_epochs=50, train_epochs=30, tolerance=0.005):
        """Full training pipeline."""
        self.pretrain(epochs=pre_epochs)
        self.initialize_clusters()

        previous_predicted_labels = None

        for epoch in range(train_epochs):
            self.train_epoch()
            predicted_labels, true_labels, acc, ari, nmi = self.evaluate()

            print(f"{epoch+1}: ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")

            num_changed = np.sum(predicted_labels != previous_predicted_labels)
            change_ratio = num_changed / len(predicted_labels)

            print(f"Change ratio: {change_ratio*100:.2f}%")

            if change_ratio <= tolerance:
                print(f"Early stopping triggered. Change ratio {change_ratio*100:.2f}% <= tolerance {tolerance*100}%")
                break

            previous_predicted_labels = predicted_labels.copy()

        print("Training Finished")

    def evaluate(self):
        """Evaluate clustering performance."""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.train_dataloader:
                x = x.to(self.device)
                q, _, _ = self.model(x)
                all_preds.append(q.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        preds = np.vstack(all_preds)
        true_labels = np.concatenate(all_labels)

        predicted_labels = np.argmax(preds, axis=1)

        acc = self.cluster_accuracy(true_labels, predicted_labels)
        ari = adjusted_rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)

        return predicted_labels, true_labels, acc, ari, nmi

    def cluster_accuracy(self, y_true, y_pred):
        """Compute clustering accuracy using Hungarian algorithm."""
        y_true = y_true.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) / y_pred.size

    def get_embeddings(self):
        """Get embeddings for all samples."""
        self.model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for x, y in self.train_dataloader:
                x = x.to(self.device)
                z, _ = self.model.autoencoder(x)
                all_embeddings.append(z.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        embeddings = np.vstack(all_embeddings)
        labels = np.concatenate(all_labels)
        return embeddings, labels
