# BioBatchNet

[![PyPI version](https://badge.fury.io/py/biobatchnet.svg)](https://badge.fury.io/py/biobatchnet)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

BioBatchNet is a deep learning framework for batch effect correction in biological data, supporting both single-cell RNA-seq (scRNA-seq) and Imaging Mass Cytometry (IMC) data.

## Installation

### From PyPI (Recommended)
```bash
pip install biobatchnet
```

### From Source
```bash
git clone https://github.com/UoM-HealthAI/BioBatchNet
cd BioBatchNet
pip install -e .
```

### Prerequisites
```bash
conda create -n biobatchnet python=3.10
conda activate biobatchnet
pip install torch numpy pandas anndata
```

## Quick Start

### Basic Usage

```python
import BioBatchNet
from BioBatchNet import correct_batch_effects
import anndata as ad

# Load your data
adata = ad.read_h5ad('your_data.h5ad')
X = adata.X
batch_labels = adata.obs['batch'].values

# Correct batch effects; returns (bio_embeddings, batch_embeddings)
bio_embeddings, batch_embeddings = correct_batch_effects(
    data=X,
    batch_info=batch_labels,
    data_type='imc',  # or 'scrna' for single-cell RNA-seq
    epochs=100
)

# Add embeddings to AnnData (recommended usage)
adata.obsm['X_biobatchnet'] = bio_embeddings
adata.obsm['X_batch'] = batch_embeddings

# Checkpoint storage: by default, training artefacts are written to a temporary
# directory and deleted automatically. To keep checkpoints, pass
# save_dir='path/to/output'.
```

## Advanced Usage

### Custom Loss Weights

For IMC data with specific batch characteristics:

```python
# Define custom loss weights
loss_weights = {
    'recon_loss': 10,        # Reconstruction loss weight
    'discriminator': 0.3,    # Adversarial loss weight
    'classifier': 1,         # Batch classifier loss weight
    'mmd_loss_1': 0,        # MMD loss weight
    'kl_loss_1': 0.005,     # KL divergence weight for bio encoder
    'kl_loss_2': 0.1,       # KL divergence weight for batch encoder
    'ortho_loss': 0.01      # Orthogonality loss weight
}

bio_embeddings, batch_embeddings = correct_batch_effects(
    data=X,
    batch_info=batch_labels,
    data_type='imc',
    latent_dim=20,
    epochs=100,
    loss_weights=loss_weights
)
```

### Custom Architecture Parameters

Fine-tune the neural network architecture:

```python
bio_embeddings, batch_embeddings = correct_batch_effects(
    data=X,
    batch_info=batch_labels,
    data_type='imc',
    latent_dim=20,
    epochs=100,
    bio_encoder_hidden_layers=[500, 2000, 2000],
    batch_encoder_hidden_layers=[500],
    decoder_hidden_layers=[2000, 2000, 500],
    batch_classifier_layers_power=[500, 2000, 2000],
    batch_classifier_layers_weak=[128]
)
```

### Direct Model Access

For more control, use the models directly:

```python
from BioBatchNet import IMCVAE, GeneVAE

# For IMC data
model = IMCVAE(
    in_sz=40,           # Number of features
    out_sz=40,          # Output dimension
    num_batch=4,        # Number of batches
    latent_sz=20,       # Latent dimension
    bio_encoder_hidden_layers=[500, 2000, 2000],
    batch_encoder_hidden_layers=[500],
    decoder_hidden_layers=[2000, 2000, 500],
    batch_classifier_layers_power=[500, 2000, 2000],
    batch_classifier_layers_weak=[128]
)

# Train the model
model.fit(data, batch_labels, epochs=100, lr=1e-3)

# Get biological embeddings (batch-corrected representations)
bio_embeddings = model.get_bio_embeddings(data)

# Get corrected data
corrected_data = model.correct_batch_effects(data)
```

## Data Formats

### Config-Driven Training Scripts

Scripts such as `BioBatchNet/imc.py` and `BioBatchNet/scrna.py` reproduce our research training pipeline. They expect the original datasets to be placed under `Data/...` (see config YAMLs for paths) and are not included in the pip distribution. For typical usage please prefer the Python API or model classes above.

### Input Data Requirements

1. **Data Matrix**: 
   - NumPy array or PyTorch tensor
   - Shape: (n_cells, n_features)
   - For IMC: Antibody expression matrix

2. **Batch Information**:
   - NumPy array or list
   - Can be string labels (e.g., ['Patient1', 'Patient2']) or numeric
   - Length must match number of cells

### Output

- **Corrected Data**: Same shape as input, with batch effects removed
- **Bio Embeddings**: Low-dimensional biological representations (n_cells, latent_dim)

## Recommended Parameters

### For IMC Data

```python
# Small dataset (< 10 batches)
loss_weights = {
    'recon_loss': 10,
    'discriminator': 0.3,
    'classifier': 1,
    'mmd_loss_1': 0,
    'kl_loss_1': 0.005,
    'kl_loss_2': 0.1,
    'ortho_loss': 0.01
}

# Large dataset (> 30 batches)
loss_weights = {
    'recon_loss': 10,
    'discriminator': 0.1,
    'classifier': 1,
    'mmd_loss_1': 0.01,
    'kl_loss_1': 0.0,
    'kl_loss_2': 0.1,
    'ortho_loss': 0.01
}
```

## Example Workflow

```python
import BioBatchNet
import anndata as ad
import numpy as np
from BioBatchNet import correct_batch_effects

# 1. Load data
adata = ad.read_h5ad('IMMUcan_batch.h5ad')
print(f"Data shape: {adata.shape}")
print(f"Batches: {adata.obs['BATCH'].unique()}")

# 2. Prepare data
X = adata.X
batch_labels = adata.obs['BATCH'].values

# Convert categorical to numpy array if needed
if hasattr(batch_labels, 'to_numpy'):
    batch_labels = batch_labels.to_numpy()

# 3. Correct batch effects
corrected = correct_batch_effects(
    data=X,
    batch_info=batch_labels,
    data_type='imc',
    latent_dim=20,
    epochs=100
)

# 4. Store results
adata.layers['corrected'] = corrected

# 5. Optional: Get embeddings for visualization
from BioBatchNet import IMCVAE
model = IMCVAE(
    in_sz=X.shape[1],
    out_sz=X.shape[1],
    num_batch=len(np.unique(batch_labels)),
    latent_sz=20
)
model.fit(X, batch_labels, epochs=100)
embeddings = model.get_bio_embeddings(X)
adata.obsm['X_biobatchnet'] = embeddings

# 6. Visualize results (using scanpy)
import scanpy as sc
sc.pp.neighbors(adata, use_rep='X_biobatchnet')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['BATCH', 'celltype'])
```

## IMC Training Tips and Best Practices

1. **Data Preprocessing**: 
   - For IMC: post-processing is typically unnecessary.

2. **Batch Size**:
   - Default batch size is 256
   - Reduce if encountering memory issues

3. **Number of Epochs**:
   - Start with 100 epochs for initial testing
   - Monitor loss convergence

4. **Latent Dimension**:
   - Default is 20
   - Increase for complex datasets with many cell types
   - Decrease for simpler datasets

## Troubleshooting

### NAN Issues
```python
# Use smaller learning rate
model.fit(data, batch_labels, lr=1e-4)  # Lower learning rate
```


## Features
- **Multi-modal Support**: Works with both scRNA-seq and IMC data
- **Easy-to-Use API**: One-line batch correction function
- **Flexible Architecture**: Customizable neural network parameters
- **Adaptive Loss Weights**: Automatically adjusts based on dataset characteristics
- **Comprehensive Documentation**: Detailed usage examples and best practices

## Citation

If you use BioBatchNet in your research, please cite:

```
Liu H, Zhang S, Mao S, et al. BioBatchNet: A Dual-Encoder Framework for Robust Batch Effect Correction in Imaging Mass Cytometry[J]. bioRxiv, 2025: 2025.03. 15.643447.
```

## License

MIT License
