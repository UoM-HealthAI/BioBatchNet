import torch
import torch.nn as nn
import pytorch_lightning as pl

from .models.model import IMCVAE, GeneVAE
from .utils.loss import kl_divergence, orthogonal_loss, ZINBLoss
from .config import Config


class BioBatchNetModule(pl.LightningModule):
    """Lightning module for batch effect correction."""

    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config

        # Model
        if config.mode == 'imc':
            self.model = IMCVAE(config.model)
        else:
            self.model = GeneVAE(config.model)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.zinb_loss = ZINBLoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Loss weights
        self.lw = config.loss

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        data, batch_id, *_ = batch
        out = self(data)

        # Reconstruction loss
        if self.config.mode == 'imc':
            recon_loss = self.mse_loss(data, out.reconstruction)
        else:
            recon_loss = self.zinb_loss(data, out.mean, out.disp, out.pi)

        # KL losses
        kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
        kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()
        kl_size = 0
        
        if self.config.mode == 'rna':
            kl_size = kl_divergence(out.size_mu, out.size_logvar).mean()

        # Classification losses
        disc_loss = self.ce_loss(out.bio_batch_pred, batch_id)
        clf_loss = self.ce_loss(out.batch_batch_pred, batch_id)

        # Orthogonal loss
        ortho_loss = orthogonal_loss(out.bio_z, out.batch_z)

        # Total loss
        loss = (
            self.lw.recon * recon_loss +
            self.lw.discriminator * disc_loss +
            self.lw.classifier * clf_loss +
            self.lw.kl_bio * kl_bio +
            self.lw.kl_batch * kl_batch +
            self.lw.ortho * ortho_loss +
            (self.lw.kl_size * kl_size if self.config.mode == 'rna' else 0)
        )

        # Accuracy
        bio_acc = (out.bio_batch_pred.argmax(1) == batch_id).float().mean()
        batch_acc = (out.batch_batch_pred.argmax(1) == batch_id).float().mean()

        # Logging
        self.log('loss', loss, prog_bar=True)
        self.log('recon', recon_loss)
        self.log('disc', disc_loss)
        self.log('clf', clf_loss)
        self.log('kl_bio', kl_bio)
        self.log('kl_batch', kl_batch)
        self.log('ortho', ortho_loss)
        self.log('bio_acc', bio_acc, prog_bar=True)
        self.log('batch_acc', batch_acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        tc = self.config.trainer
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=tc.scheduler_step,
            gamma=tc.scheduler_gamma,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_embeddings(self, dataloader):
        """Extract embeddings from data."""
        self.eval()
        bio_z_list, batch_z_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                out = self(data)
                bio_z_list.append(out.bio_z.cpu())
                batch_z_list.append(out.batch_z.cpu())

        return torch.cat(bio_z_list).numpy(), torch.cat(batch_z_list).numpy()
