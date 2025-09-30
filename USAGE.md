# BioBatchNet Usage Guide

BioBatchNet is a VAE framework for batch effect correction in biological data, supporting both **Imaging Mass Cytometry (IMC)** and **single-cell RNA-seq (scRNA-seq)** data.

---

## Installation

### Create Environment

```bash
conda create -n biobatchnet python=3.10
conda activate biobatchnet
```

### Install Dependencies

```bash
pip install torch>=2.0.0 numpy>=1.26.0 pandas>=2.2.0 scikit-learn>=1.3.0 \
            scipy>=1.10.0 tqdm>=4.65.0 pyyaml>=6.0 anndata>=0.10.0 \
            scanpy>=1.9.0 matplotlib>=3.7.0 seaborn>=0.12.0 h5py>=3.8.0
```

Or use the provided requirements file:

```bash
pip install -r requirements.txt
```

### Install BioBatchNet

```bash
git clone https://github.com/Manchester-HealthAI/BioBatchNet
cd BioBatchNet
pip install -e .
```

---

## Quick Start

### Download Example Data

```python
import gdown
import anndata as ad
import numpy as np
import pandas as pd

# Download IMMUcan IMC dataset from Google Drive
FILE_ID = "1S0AgcT0J7tnRtnnshRzAkECwhse0mTrK"
gdown.download(id=FILE_ID, output="IMMUcan_batch.h5ad", quiet=False)

# Load data
adata = ad.read_h5ad("IMMUcan_batch.h5ad")
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Extract and convert batch labels
unique_batches = np.unique(adata.obs['BATCH'].values)
batch_to_int = {batch: i for i, batch in enumerate(unique_batches)}
batch_labels = np.array([batch_to_int[b] for b in adata.obs['BATCH'].values])

print(f"Data: {X.shape[0]:,} cells, {X.shape[1]} features, {len(unique_batches)} batches")
```

### Basic Batch Correction

```python
from biobatchnet import correct_batch_effects

# Run batch correction with default parameters
bio_embeddings, batch_embeddings = correct_batch_effects(
    data=pd.DataFrame(X),
    batch_info=pd.DataFrame({'BATCH': batch_labels}),
    batch_key='BATCH',
    data_type='imc',        # 'imc' or 'scrna'
    latent_dim=20,          # Latent space dimension
    epochs=100,             # Training epochs
    device='cuda'           # or 'cpu'
)

# Add embeddings to AnnData object
adata.obsm['X_biobatchnet'] = bio_embeddings

print(f"✓ Biological embeddings: {bio_embeddings.shape}")
```

### Visualize Results

```python
import scanpy as sc
import matplotlib.pyplot as plt

# UMAP visualization function
def plot_umap(adata, use_rep="X", title=None, color_by=['BATCH', 'celltype']):
    adata_vis = adata.copy()
    adata_vis.obs['BATCH'] = adata_vis.obs['BATCH'].astype("category")
    sc.pp.neighbors(adata_vis, use_rep=use_rep)
    sc.tl.umap(adata_vis)
    sc.pl.umap(adata_vis, color=color_by, title=title, frameon=False, wspace=0.5)

# Visualize corrected data
plot_umap(adata, use_rep="X_biobatchnet", title="After BioBatchNet Correction")
```

---

## Method 1: Simple API (Recommended)

The easiest way to use BioBatchNet with automatic parameter selection.

```python
from biobatchnet import correct_batch_effects

bio_embeddings, batch_embeddings = correct_batch_effects(
    data=pd.DataFrame(X),
    batch_info=pd.DataFrame({'BATCH': batch_labels}),
    batch_key='BATCH',
    data_type='imc',
    latent_dim=20,
    epochs=100
)
```

**Parameters:**
- `data`: Expression matrix as pandas DataFrame
- `batch_info`: Batch labels as pandas DataFrame
- `batch_key`: Column name for batch labels (default: 'BATCH')
- `data_type`: 'imc' or 'scrna'
- `latent_dim`: Latent space dimension (15-25 for IMC, 20-50 for scRNA-seq)
- `epochs`: Training epochs (100-200 recommended)

---

## Method 2: Custom Loss Weights

Fine-tune training objectives by adjusting loss weights.

```python
# Define custom loss weights
custom_loss_weights = {
    'recon_loss': 10,       # Reconstruction loss (default: 10)
    'discriminator': 0.1,   # Batch mixing (default: 0.3, lower = more mixing)
    'classifier': 1.0,      # Batch retention (default: 1)
    'kl_loss_1': 0.005,     # KL divergence for bio encoder
    'kl_loss_2': 0.1,       # KL divergence for batch encoder
    'ortho_loss': 0.01      # Orthogonality constraint (default: 0.01)
}

bio_embeddings, _ = correct_batch_effects(
    data=pd.DataFrame(X),
    batch_info=pd.DataFrame({'BATCH': batch_labels}),
    batch_key='BATCH',
    data_type='imc',
    latent_dim=20,
    epochs=100,
    loss_weights=custom_loss_weights
)
```

### Loss Weight Guidelines

| Parameter | Default | Description | When to Adjust |
|-----------|---------|-------------|----------------|
| `recon_loss` | 10 | Reconstruction quality | Increase for better reconstruction |
| `discriminator` | 0.3 | Batch mixing strength | Lower for many batches (0.1-0.2) |
| `classifier` | 1 | Batch info retention | Keep default |
| `kl_loss_1` | 0.005 | Bio encoder regularization | Increase if overfitting |
| `kl_loss_2` | 0.1 | Batch encoder regularization | Keep default |
| `ortho_loss` | 0.01 | Bio/batch orthogonality | Keep default |

**Recommended presets:**

```python
# For small datasets (< 10 batches)
loss_weights = {
    'recon_loss': 10,
    'discriminator': 0.3,
    'classifier': 1,
    'kl_loss_1': 0.005,
    'kl_loss_2': 0.1,
    'ortho_loss': 0.01
}

# For large datasets (15+ batches)
loss_weights = {
    'recon_loss': 10,
    'discriminator': 0.1,
    'classifier': 1,
    'kl_loss_1': 0.0,
    'kl_loss_2': 0.1,
    'ortho_loss': 0.01
}
```

---

## Method 3: Direct Model Usage (Advanced)

For full control over model architecture and training.

### IMC Data

```python
from biobatchnet import IMCVAE

# Create model with custom architecture
n_cells, n_features = X.shape
n_batches = len(np.unique(batch_labels))

model = IMCVAE(
    in_sz=n_features,
    out_sz=n_features,
    latent_sz=20,
    num_batch=n_batches,
    bio_encoder_hidden_layers=[256, 512, 512],        # Custom bio encoder
    batch_encoder_hidden_layers=[128, 256],           # Custom batch encoder
    decoder_hidden_layers=[512, 512, 256],            # Custom decoder
    batch_classifier_layers_power=[500, 2000, 2000],  # Discriminator
    batch_classifier_layers_weak=[128]                # Batch classifier
)

# Train model
custom_loss = {
    'recon_loss': 12,
    'discriminator': 0.25,
    'classifier': 1.2,
    'kl_loss_1': 0.005,
    'kl_loss_2': 0.1,
    'ortho_loss': 0.015
}

model.fit(
    data=X,
    batch_info=batch_labels,
    epochs=100,
    lr=1e-4,
    batch_size=256,
    loss_weights=custom_loss,
    device='cuda'
)

# Extract embeddings
bio_embeddings = model.get_bio_embeddings(X)

# Or get both bio and batch embeddings
bio_emb, batch_emb = model.correct_batch_effects(X)
```

### scRNA-seq Data

```python
from biobatchnet import GeneVAE

model = GeneVAE(
    in_sz=n_features,
    out_sz=n_features,
    latent_sz=30,
    num_batch=n_batches,
    bio_encoder_hidden_layers=[512, 1024, 1024],
    batch_encoder_hidden_layers=[256],
    decoder_hidden_layers=[1024, 1024, 512],
    batch_classifier_layers_power=[512, 1024, 1024],
    batch_classifier_layers_weak=[128]
)

model.fit(data=X, batch_info=batch_labels, epochs=200, lr=1e-4)
bio_embeddings = model.get_bio_embeddings(X)
```

---

## Parameter Reference

### Model Architecture

| Parameter | Default (IMC) | Description |
|-----------|---------------|-------------|
| `bio_encoder_hidden_layers` | [512, 1024, 1024] | Bio encoder layer sizes |
| `batch_encoder_hidden_layers` | [256] | Batch encoder layer sizes |
| `decoder_hidden_layers` | [1024, 1024, 512] | Decoder layer sizes |
| `batch_classifier_layers_power` | [512, 1024, 1024] | Discriminator architecture |
| `batch_classifier_layers_weak` | [128] | Batch classifier architecture |

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 100 | Training epochs (100-200 recommended) |
| `lr` | 1e-4 | Learning rate (1e-4 to 1e-3) |
| `batch_size` | 256 | Training batch size (reduce if OOM) |
| `device` | 'cuda' | 'cuda' or 'cpu' |

---

## Data Format Requirements

### Input Data

1. **Expression Matrix**
   - Type: NumPy array, pandas DataFrame, or sparse matrix
   - Shape: `(n_cells, n_features)`
   - For scRNA-seq: gene counts or normalized expression
   - For IMC: protein expression values

2. **Batch Labels**
   - Type: NumPy array, pandas Series, or DataFrame
   - Length: Must match number of cells
   - Format: Can be strings or integers
   - **Important**: Convert categorical string labels to integers before passing to model

### Output

- **Bio Embeddings**: `(n_cells, latent_dim)` - Batch-corrected biological representations
- **Batch Embeddings**: `(n_cells, latent_dim)` - Batch-specific information

---

## Complete Workflow Example

```python
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from biobatchnet import correct_batch_effects

# 1. Load data
adata = ad.read_h5ad('your_data.h5ad')
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# 2. Prepare batch labels (convert to integers)
unique_batches = np.unique(adata.obs['BATCH'].values)
batch_to_int = {batch: i for i, batch in enumerate(unique_batches)}
batch_labels = np.array([batch_to_int[b] for b in adata.obs['BATCH'].values])

# 3. Run batch correction
bio_embeddings, batch_embeddings = correct_batch_effects(
    data=pd.DataFrame(X),
    batch_info=pd.DataFrame({'BATCH': batch_labels}),
    batch_key='BATCH',
    data_type='imc',  # or 'scrna'
    latent_dim=20,
    epochs=100,
    device='cuda'
)

# 4. Store results
adata.obsm['X_biobatchnet'] = bio_embeddings
adata.obsm['X_batch'] = batch_embeddings

# 5. Visualize
sc.pp.neighbors(adata, use_rep='X_biobatchnet')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['BATCH', 'celltype'])

# 6. Save corrected data
adata.write('corrected_data.h5ad')


**Training instability:**
```python
model.fit(data=X, batch_info=batch_labels, lr=5e-5)  # Lower learning rate
```

---

## Citation

If you use BioBatchNet in your research, please cite:

```
[Citation to be added upon publication]
```

---

## Support

- **GitHub Issues**: [https://github.com/Manchester-HealthAI/BioBatchNet/issues](https://github.com/Manchester-HealthAI/BioBatchNet/issues)
- **Tutorial Notebook**: See `tutorial.ipynb` for interactive examples
- **API Documentation**: See `README.md`

---

## License

MIT License