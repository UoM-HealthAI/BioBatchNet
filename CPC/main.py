import argparse
import os
import numpy as np
import torch
import scanpy as sc
from torch.utils.data import DataLoader

from model import ConstrainedClustering
from dataset import GeneralDataset, MLDataset, CLDataset
from trainer import Trainer
from utils import generate_random_pair, transitive_closure


def main():
    parser = argparse.ArgumentParser(description='Constrained Pairwise Clustering')
    parser.add_argument('--data_path', '-d', type=str, required=True, help='Path to adata file (.h5ad)')
    parser.add_argument('--label_key', type=str, default='celltype', help='Key in adata.obs for labels')
    parser.add_argument('--obsm_key', type=str, default='X_biobatchnet', help='Key in adata.obsm for embeddings')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters')
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent dimension')
    parser.add_argument('--num_constraints', type=int, default=4000, help='Number of constraints')
    parser.add_argument('--pre_epochs', type=int, default=50, help='Pretraining epochs')
    parser.add_argument('--train_epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Load adata
    print(f"Loading adata from {args.data_path}")
    adata = sc.read_h5ad(args.data_path)

    # Get input dimension
    input_dim = adata.obsm[args.obsm_key].shape[1]

    # Get labels for constraint generation
    labels = adata.obs[args.label_key]
    if hasattr(labels, 'cat'):
        labels = labels.cat.codes.values
    elif not np.issubdtype(labels.dtype, np.number):
        from sklearn.preprocessing import LabelEncoder
        labels = LabelEncoder().fit_transform(labels.values)
    else:
        labels = labels.values

    # Generate constraints
    print(f"Generating {args.num_constraints} constraints")
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = generate_random_pair(labels, args.num_constraints * 2)
    ml_ind1, ml_ind2, cl_ind1, cl_ind2 = transitive_closure(ml_ind1, ml_ind2, cl_ind1, cl_ind2, len(adata))
    ml_ind1, ml_ind2 = ml_ind1[:args.num_constraints], ml_ind2[:args.num_constraints]
    cl_ind1, cl_ind2 = cl_ind1[:args.num_constraints], cl_ind2[:args.num_constraints]

    # Create datasets
    train_dataset = GeneralDataset(adata, args.label_key, args.obsm_key)
    ml_dataset = MLDataset(ml_ind1, ml_ind2, adata, args.obsm_key)
    cl_dataset = CLDataset(cl_ind1, cl_ind2, adata, args.obsm_key)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    ml_loader = DataLoader(ml_dataset, batch_size=128, shuffle=True)
    cl_loader = DataLoader(cl_dataset, batch_size=128, shuffle=True)

    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ConstrainedClustering(input_dim=input_dim, latent_dim=args.latent_dim, n_clusters=args.n_clusters)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer and train
    trainer = Trainer(model, optimizer, train_loader, ml_loader, cl_loader, device)
    trainer.train(pre_epochs=args.pre_epochs, train_epochs=args.train_epochs)

    # Final evaluation
    predicted_labels, true_labels, acc, ari, nmi = trainer.evaluate()
    print(f"\nFinal: ACC={acc:.4f}, ARI={ari:.4f}, NMI={nmi:.4f}")

    # Get embeddings with shuffle=False to preserve order
    eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for x, _ in eval_loader:
            x = x.to(device)
            z, _ = model.autoencoder(x)
            all_embeddings.append(z.cpu().numpy())
    embeddings = np.vstack(all_embeddings)
    adata.obsm['X_cpc'] = embeddings

    # Save adata
    save_dir = os.path.join(os.path.dirname(__file__), 'saved')
    os.makedirs(save_dir, exist_ok=True)

    # Use original filename
    filename = os.path.basename(args.data_path).replace('.h5ad', '_cpc.h5ad')
    save_path = os.path.join(save_dir, filename)

    adata.write_h5ad(save_path)
    print(f"Saved adata with X_cpc to {save_path}")


if __name__ == '__main__':
    main()
