# BioBatchNet

[![PyPI version](https://badge.fury.io/py/biobatchnet.svg)](https://badge.fury.io/py/biobatchnet)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A dual-encoder VAE framework for batch effect correction in **single-cell RNA-seq** and **Imaging Mass Cytometry (IMC)** data.

---

## Installation

```bash
pip install biobatchnet
```

For development:

```bash
git clone https://github.com/UoM-HealthAI/BioBatchNet
cd BioBatchNet
pip install -e .
```

---

## Quick Start

```python
import scanpy as sc
from biobatchnet import correct_batch_effects

adata = sc.read_h5ad('your_data.h5ad')

bio_emb, batch_emb = correct_batch_effects(
    adata,
    batch_key='BATCH',
    data_type='imc',    # 'imc' or 'seq'
)

adata.obsm['X_biobatchnet'] = bio_emb

# Visualize
sc.pp.neighbors(adata, use_rep='X_biobatchnet')
sc.tl.umap(adata)
sc.pl.umap(adata, color=['BATCH', 'celltype'])
```

### Custom Loss Weights

```python
bio_emb, batch_emb = correct_batch_effects(
    adata,
    batch_key='BATCH',
    data_type='imc',
    loss_weights={'discriminator': 0.1},  # lower = stronger batch mixing
)
```

See [USAGE.md](USAGE.md) for full parameter reference and [tutorial.ipynb](tutorial.ipynb) for an interactive walkthrough.

---

## Config-based Training

For reproducing experiments with preset hyperparameters:

```bash
python -m biobatchnet.train --config immucan --data path/to/IMMUcan.h5ad
```

Available presets: `damond`, `hoch`, `immucan`, `pancreas`, `macaque`, `lung`, `mousebrain`

Override loss weights via CLI:

```bash
python -m biobatchnet.train --config pancreas --data path/to/pancreas.h5ad \
    --loss.discriminator 0.1 --loss.kl_bio 0.001
```

---

## Data

- **IMC datasets**: [Bodenmiller Group IMC datasets](https://github.com/BodenmillerGroup/imcdatasets)
- **scRNA-seq datasets**: [Google Drive](https://drive.google.com/drive/folders/1m4AkNc_KMadp7J_lL4jOQj9DdyKutEZ5?usp=drive_link)

---

## Citation

If you use BioBatchNet in your research, please cite:

```
Liu H, Zhang S, Mao S, et al. BioBatchNet: A Dual-Encoder Framework for Robust Batch Effect
Correction in Imaging Mass Cytometry[J]. bioRxiv, 2025: 2025.03.15.643447.
```

---

## License

MIT License
