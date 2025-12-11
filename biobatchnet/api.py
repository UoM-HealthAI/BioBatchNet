"""Simple API for batch effect correction."""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .models.model import IMCVAE, GeneVAE
from .config import Config, ModelConfig, LossConfig, TrainerConfig
from .utils.trainer import Trainer
from .utils.user_dataset import UserIMCDataset


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
    save_dir='./saved',
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
        device: 'cuda' or 'cpu'
        save_dir: directory to save checkpoints

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
        save_dir=save_dir,
    )

    config = Config(
        name='api_training',
        mode=data_type,
        model=model_config,
        loss=loss_config,
        trainer=trainer_config,
    )

    # Create model
    if data_type == 'imc':
        model = IMCVAE(model_config)
    else:
        model = GeneVAE(model_config)

    # Create dataset and dataloader
    dataset = UserIMCDataset(data, batch_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        print("CUDA not available, using CPU")

    # Train
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        device=device,
    )
    trainer.train()

    # Get embeddings
    bio_embeddings, batch_embeddings = model.get_embeddings(data)

    return bio_embeddings, batch_embeddings
