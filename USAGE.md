# BioBatchNet Usage Guide

## Installation

```bash
pip install biobatchnet
```

Or install from source:

```bash
git clone https://github.com/UoM-HealthAI/BioBatchNet
cd BioBatchNet
pip install -e .
```

---

## Python API

### Basic Usage

```python
import scanpy as sc
from biobatchnet import correct_batch_effects

adata = sc.read_h5ad('your_data.h5ad')

bio_emb, batch_emb = correct_batch_effects(
    adata,
    batch_key='BATCH',
    data_type='imc',
)

adata.obsm['X_biobatchnet'] = bio_emb
```

### Full Parameters

```python
bio_emb, batch_emb = correct_batch_effects(
    adata,                      # AnnData object
    batch_key='BATCH',          # obs column for batch labels
    cell_type_key='celltype',   # optional: obs column for cell types
    data_type='imc',            # 'imc' or 'seq'
    latent_dim=20,              # latent space dimension
    epochs=100,                 # max training epochs
    lr=1e-4,                    # learning rate
    batch_size=128,             # training batch size
    device='auto',              # 'auto', 'cuda', or 'cpu'
    loss_weights=None,          # optional: override default loss weights
)
```

### Custom Loss Weights

Pass a dict to `loss_weights` to override defaults:

```python
bio_emb, batch_emb = correct_batch_effects(
    adata,
    batch_key='BATCH',
    data_type='imc',
    loss_weights={
        'discriminator': 0.1,   # batch mixing strength (lower = more mixing)
        'kl_bio': 0.01,         # bio encoder regularization
    },
)
```

| Parameter | IMC default | scRNA-seq default | Description |
|-----------|-------------|-------------------|-------------|
| `recon` | 10.0 | 10.0 | Reconstruction quality |
| `discriminator` | 0.3 | 0.04 | Batch mixing strength |
| `classifier` | 1.0 | 1.0 | Batch info retention |
| `kl_bio` | 0.005 | 1e-6 | Bio encoder regularization |
| `kl_batch` | 0.1 | 0.01 | Batch encoder regularization |
| `ortho` | 0.01 | 0.0002 | Bio/batch orthogonality |
| `kl_size` | - | 0.002 | Size factor regularization (seq only) |

---

## Config-based Training

For reproducible experiments using preset hyperparameters:

```bash
python -m biobatchnet.train --config <preset> --data <path_to_h5ad>
```

### Available Presets

| Preset | Data type | Epochs |
|--------|-----------|--------|
| `damond` | IMC | 30 |
| `hoch` | IMC | 30 |
| `immucan` | IMC | 30 |
| `pancreas` | scRNA-seq | 50 |
| `macaque` | scRNA-seq | 50 |
| `lung` | scRNA-seq | 100 |
| `mousebrain` | scRNA-seq | 50 |

### Override Parameters

```bash
python -m biobatchnet.train --config immucan --data data.h5ad \
    --loss.discriminator 0.1 \
    --loss.kl_bio 0.01 \
    --trainer.epochs 50 \
    --seed 42
```

---

## Data Format

**Input**: AnnData object (`.h5ad` file) with:
- `adata.X`: expression matrix (cells x features), dense or sparse
- `adata.obs[batch_key]`: batch labels (string or integer)
- `adata.obs[cell_type_key]` (optional): cell type annotations

**Output**:
- `bio_embeddings`: `(n_cells, latent_dim)` — batch-corrected representations
- `batch_embeddings`: `(n_cells, latent_dim)` — batch-specific information

---

## Support

- **GitHub Issues**: [https://github.com/UoM-HealthAI/BioBatchNet/issues](https://github.com/UoM-HealthAI/BioBatchNet/issues)
- **Tutorial**: See `tutorial.ipynb`

---

## License

MIT License
