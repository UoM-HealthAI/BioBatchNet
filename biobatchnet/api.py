import torch
import numpy as np
import pandas as pd
from .models.model import IMCVAE, GeneVAE

def correct_batch_effects(
    data,
    batch_info,
    batch_key='batch_id',
    data_type='imc',
    latent_dim=20,
    epochs=100,
    loss_weights=None,
    save_dir=None,
    **kwargs,
):
    """
    Simple API for batch effect correction
    
    Args:
        data: pandas DataFrame or numpy array (cells x features)
        batch_info: pandas DataFrame with batch information
        batch_key: column name for batch labels
        data_type: 'imc' or 'scrna'
        latent_dim: latent space dimension
        epochs: training epochs
        loss_weights: dict of loss weights, e.g. {
            'recon_loss': 10,
            'discriminator': 0.3,
            'classifier': 1,
            'kl_loss_1': 0.005,
            'kl_loss_2': 0.1,
            'ortho_loss': 0.01
        }
        save_dir: optional path to store training checkpoints; defaults to a
            temporary directory that is cleaned up automatically.
        
    Returns:
        numpy array: corrected data
    """
    if isinstance(data, pd.DataFrame):
        data = data.values
    
    if isinstance(batch_info, pd.DataFrame):
        batch_labels = batch_info[batch_key].values
    else:
        batch_labels = batch_info
    
    # Convert string batch labels to integers
    unique_batches = np.unique(batch_labels)
    batch_to_int = {batch: i for i, batch in enumerate(unique_batches)}
    batch_labels = np.array([batch_to_int[batch] for batch in batch_labels])
    
    num_batches = len(unique_batches)
    input_dim = data.shape[1]
    
    # Default loss weights
    if loss_weights is None:
        if data_type == 'imc':
            if num_batches >= 15:
                loss_weights = {
                    'recon_loss': 10, 'discriminator': 0.1, 'classifier': 1,
                    'kl_loss_1': 0.005, 'kl_loss_2': 0.1, 'ortho_loss': 0.01
                }
            else:
                loss_weights = {
                    'recon_loss': 10, 'discriminator': 0.3, 'classifier': 1,
                    'kl_loss_1': 0.005, 'kl_loss_2': 0.1, 'ortho_loss': 0.01
                }
        else:  # scrna
            loss_weights = {
                'recon_loss': 10, 'discriminator': 0.04, 'classifier': 1,
                'kl_loss_1': 1e-7, 'kl_loss_2': 0.01, 'ortho_loss': 0.0002, 'kl_loss_size': 0.002
            }
    
    if data_type == 'imc':
        # Extract architecture params from kwargs or use defaults
        bio_encoder_layers = kwargs.pop('bio_encoder_hidden_layers', [512, 1024, 1024])
        batch_encoder_layers = kwargs.pop('batch_encoder_hidden_layers', [256])
        decoder_layers = kwargs.pop('decoder_hidden_layers', [1024, 1024, 512])
        classifier_power = kwargs.pop('batch_classifier_layers_power', [512, 1024, 1024])
        classifier_weak = kwargs.pop('batch_classifier_layers_weak', [128])
        
        model = IMCVAE(
            in_sz=input_dim,
            out_sz=input_dim,
            latent_sz=latent_dim,
            num_batch=num_batches,
            bio_encoder_hidden_layers=bio_encoder_layers,
            batch_encoder_hidden_layers=batch_encoder_layers,
            decoder_hidden_layers=decoder_layers,
            batch_classifier_layers_power=classifier_power,
            batch_classifier_layers_weak=classifier_weak,
            **kwargs
        )
    else: 
        model = GeneVAE(
            in_sz=input_dim,
            out_sz=input_dim,
            latent_sz=latent_dim,
            num_batch=num_batches,
            bio_encoder_hidden_layers=[500, 2000, 2000],
            batch_encoder_hidden_layers=[500],
            decoder_hidden_layers=[2000, 2000, 500],
            batch_classifier_layers_power=[500, 2000, 2000],
            batch_classifier_layers_weak=[128],
            **kwargs
        )
    
    model.fit(
        data,
        batch_labels,
        epochs=epochs,
        loss_weights=loss_weights,
        save_dir=save_dir,
        **kwargs,
    )
    bio_embeddings, batch_embeddings = model.correct_batch_effects(data)
    
    return bio_embeddings, batch_embeddings
