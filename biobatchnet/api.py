"""Simple API for batch effect correction."""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from .config import Config, ModelConfig, LossConfig, TrainerConfig
from .module import IMCModule, RNAModule
from .utils.dataset import BBNDataset


def correct_batch_effects(
    data,
    batch_info,
    batch_key='batch_id',
    data_type='imc',
    latent_dim=20,
    epochs=100,
    lr=1e-3,
    batch_size=256,
    device='cuda',
):
    """
    Simple API for batch effect correction.

    Args:
        data: pandas DataFrame or numpy array (cells x features)
        batch_info: pandas DataFrame with batch information or array of batch labels
        batch_key: column name for batch labels (if batch_info is DataFrame)
        data_type: 'imc' or 'rna'
        latent_dim: latent space dimension
        epochs: training epochs
        lr: learning rate
        batch_size: batch size
        device: 'cuda', 'cpu', or 'auto'

    Returns:
        bio_embeddings: batch-corrected embeddings
        batch_embeddings: batch-specific embeddings
    """
    # Process input data
    if isinstance(data, pd.DataFrame):
        data = data.values

    if isinstance(batch_info, pd.DataFrame):
        batch_labels = batch_info[batch_key].values
    else:
        batch_labels = batch_info

    # Convert batch labels to integers
    unique_batches = np.unique(batch_labels)
    batch_to_int = {b: i for i, b in enumerate(unique_batches)}
    batch_labels = np.array([batch_to_int[b] for b in batch_labels])

    num_batches = len(unique_batches)
    input_dim = data.shape[1]

    # Create config
    model_config = ModelConfig(
        in_sz=input_dim,
        out_sz=input_dim,
        latent_sz=latent_dim,
        num_batch=num_batches,
    )

    if data_type == 'imc':
        loss_config = LossConfig(
            recon=10.0,
            discriminator=0.3,
            classifier=1.0,
            kl_bio=0.005,
            kl_batch=0.1,
            ortho=0.01,
        )
    else:  # rna
        loss_config = LossConfig(
            recon=10.0,
            discriminator=0.04,
            classifier=1.0,
            kl_bio=1e-7,
            kl_batch=0.01,
            ortho=0.0002,
            kl_size=0.002,
        )

    trainer_config = TrainerConfig(
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        early_stop=100,
    )

    config = Config(
        name='api_training',
        mode=data_type,
        model=model_config,
        loss=loss_config,
        trainer=trainer_config,
    )

    # Create dataset and dataloader
    dataset = BBNDataset(data, batch_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create Lightning module
    Module = IMCModule if data_type == 'imc' else RNAModule
    model = Module(config)

    # Train with Lightning
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[EarlyStopping(monitor='loss', patience=15, mode='min')],
        accelerator='auto' if device == 'auto' else device,
        devices=1,
        enable_progress_bar=True,
        logger=False,
    )
    trainer.fit(model, dataloader)

    # Get embeddings
    bio_embeddings, batch_embeddings = model.get_embeddings(dataloader)

    return bio_embeddings, batch_embeddings
