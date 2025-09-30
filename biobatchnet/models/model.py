from .modules import *
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import tempfile

from ..utils.trainer import Trainer
from ..parse_config import ConfigParser
from torch.utils.data import DataLoader
from ..utils.user_dataset import UserIMCDataset
        
class IMCVAE(nn.Module):
    def __init__(self, **args):
        super(IMCVAE, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        self.bio_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, self.latent_sz)
        self.batch_encoder = BaseEncoder(self.in_sz, self.batch_encoder_hidden_layers, self.latent_sz)
        self.decoder = BaseDecoder(2 * self.latent_sz, self.decoder_hidden_layers, self.out_sz)
        self.bio_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_power, self.num_batch)
        self.batch_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_weak, self.num_batch)
        
        self.alpha = 1
        self.grl = GRL(alpha=self.alpha)

    def forward(self, x):  
        # bio information 
        bio_z, mu1, logvar1 = self.bio_encoder(x)

        # batch information
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)

        # combine information
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # adversarial
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)

        # classifier
        batch_batch_pred = self.batch_classifier(batch_z)

        # reconstruction
        reconstruction = self.decoder(z_combine)

        return bio_z, mu1, logvar1, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction
    
    def fit(self, data, batch_info, epochs=100, lr=1e-4, batch_size=256, loss_weights=None, device='cuda', save_dir=None):
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU")
        device = torch.device(device)
        self.to(device)
        
        # Create dataset using UserIMCDataset
        dataset = UserIMCDataset(data, batch_info)
        # Move tensors to device
        dataset.data = dataset.data.to(device)
        dataset.batch_labels = dataset.batch_labels.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        temp_dir = None
        if save_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            save_dir_path = temp_dir.name
        else:
            save_dir_path = str(save_dir)

        # Create minimal config for Trainer
        config = {
            'name': 'api_training',
            'eval_sampling_seed': [42],
            'trainer': {
                'epochs': epochs,
                'save_dir': save_dir_path,
                'save_period': 100,
                'verbosity': 1,
                'early_stop': 100,
                'if_imc': True,
                'skip_intermediate_eval': True,
                'sampling_fraction': {'api_training': 1.0}
            },
            'loss_weights': loss_weights or {
                'recon_loss': 10,
                'discriminator': 0.3,
                'classifier': 1,
                'kl_loss_1': 0.005,
                'kl_loss_2': 0.1,
                'ortho_loss': 0.01
            }
        }
        
        config_parser = ConfigParser(config)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        trainer = Trainer(
            config=config_parser,
            model=self,
            optimizer=optimizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            scheduler=scheduler,
            device=device,
            seed=42
        )

        try:
            trainer.train()
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()
        
    def get_bio_embeddings(self, data):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            return bio_z.cpu().numpy()
            
    def correct_batch_effects(self, data):
        """Return bio and batch embeddings"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            batch_z, _, _ = self.batch_encoder(data)
            return bio_z.cpu().numpy(), batch_z.cpu().numpy()
        
class GeneVAE(nn.Module):
    def __init__(self, **args):
        super(GeneVAE, self).__init__()

        for key, value in args.items():
            setattr(self, key, value)

        self.bio_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, self.latent_sz)
        self.batch_encoder = BaseEncoder(self.in_sz, self.batch_encoder_hidden_layers, self.latent_sz)
        self.size_encoder = BaseEncoder(self.in_sz, self.bio_encoder_hidden_layers, 1)
        
        self.decoder = BaseDecoder(2 * self.latent_sz, self.decoder_hidden_layers, out_sz=1000)
        self.mean_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), MeanAct())  
        self.dispersion_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), DispAct())
        self.dropout_decoder = nn.Sequential(nn.Linear(1000,  self.out_sz), nn.Sigmoid())

        self.bio_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_power, self.num_batch)
        self.batch_classifier = BaseClassifier(self.latent_sz, self.batch_classifier_layers_weak, self.num_batch)

        self.alpha = 1
        self.grl = GRL(alpha=self.alpha)

    def forward(self, x): 
        # bio information 
        bio_z, mu1, logvar1 = self.bio_encoder(x)
        logvar1 = torch.clamp(logvar1, min=-5, max=5)
        size_factor, size_mu, size_logvar = self.size_encoder(x)
        # clamp size_logvar to avoid extremely large values that could lead to numerical overflow / NaNs
        size_logvar = torch.clamp(size_logvar, min=-5, max=5)

        # batch information
        batch_z, batch_mu, batch_logvar = self.batch_encoder(x)
        # clamp batch_logvar as well
        batch_logvar = torch.clamp(batch_logvar, min=-5, max=5)

        # combine information
        z_combine = torch.cat([bio_z, batch_z.detach()], dim=1)

        # adversarial
        bio_z_grl = self.grl(bio_z)
        bio_batch_pred = self.bio_classifier(bio_z_grl)
        
        # classifier
        batch_batch_pred = self.batch_classifier(batch_z)

        # zinb 
        h = self.decoder(z_combine)
        size_factor = torch.clamp(size_factor, min=-5, max=5)
        _mean = self.mean_decoder(h) * torch.exp(size_factor)
        _mean = torch.clamp(_mean, 1e-6, 1e8)

        _disp = self.dispersion_decoder(h)
        _pi = self.dropout_decoder(h)
        _pi = torch.clamp(_pi, 1e-6, 1.0 - 1e-6)

        return bio_z, mu1, logvar1, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi, size_factor, size_mu, size_logvar

    def fit(self, data, batch_info, epochs=100, lr=1e-3, batch_size=256, loss_weights=None, device='cuda', save_dir=None):
        # Set device
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
            print("CUDA not available, using CPU")
        device = torch.device(device)
        self.to(device)

        # Create dataset using UserIMCDataset
        dataset = UserIMCDataset(data, batch_info)
        # Move tensors to device
        dataset.data = dataset.data.to(device)
        dataset.batch_labels = dataset.batch_labels.to(device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        temp_dir = None
        if save_dir is None:
            temp_dir = tempfile.TemporaryDirectory()
            save_dir_path = temp_dir.name
        else:
            save_dir_path = str(save_dir)

        # Create minimal config for Trainer
        config = {
            'name': 'api_training',
            'eval_sampling_seed': [42],
            'trainer': {
                'epochs': epochs,
                'save_dir': save_dir_path,
                'save_period': 100,
                'verbosity': 1,
                'early_stop': 100,
                'if_imc': False,  # GeneVAE is for scRNA
                'skip_intermediate_eval': True,
                'sampling_fraction': {'api_training': 1.0}
            },
            'loss_weights': loss_weights or {
                'recon_loss': 10,
                'discriminator': 0.04,
                'classifier': 1,
                'kl_loss_1': 1e-7,
                'kl_loss_2': 0.01,
                'ortho_loss': 0.0002,
                'kl_loss_size': 0.002
            }
        }

        config_parser = ConfigParser(config)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        trainer = Trainer(
            config=config_parser,
            model=self,
            optimizer=optimizer,
            train_dataloader=dataloader,
            eval_dataloader=dataloader,
            scheduler=scheduler,
            device=device,
            seed=42
        )

        try:
            trainer.train()
        finally:
            if temp_dir is not None:
                temp_dir.cleanup()

    def get_bio_embeddings(self, data):
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            return bio_z.cpu().numpy()

    def correct_batch_effects(self, data):
        """Return bio and batch embeddings"""
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            if isinstance(data, np.ndarray):
                data = torch.FloatTensor(data)
            data = data.to(device)
            bio_z, _, _ = self.bio_encoder(data)
            batch_z, _, _ = self.batch_encoder(data)
            return bio_z.cpu().numpy(), batch_z.cpu().numpy()

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-3, max=1e3)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-3, max=1e3)
