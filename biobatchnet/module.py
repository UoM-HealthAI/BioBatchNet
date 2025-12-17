"""PyTorch Lightning modules for BioBatchNet."""
import torch
import torch.nn as nn
import pytorch_lightning as pl

from .models.model import IMCVAE, GeneVAE
from .utils.loss import kl_divergence, orthogonal_loss, ZINBLoss
from .config import Config


class BaseBBNModule(pl.LightningModule):
    """Base module with shared training logic."""

    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config
        self.lw = config.loss
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, loss, recon, disc, clf, kl_bio, kl_batch, ortho, bio_acc, batch_acc):
        """Log training metrics."""
        self.log('loss', loss, prog_bar=True)
        self.log('recon', recon)
        self.log('disc', disc)
        self.log('clf', clf)
        self.log('kl_bio', kl_bio)
        self.log('kl_batch', kl_batch)
        self.log('ortho', ortho)
        self.log('bio_acc', bio_acc, prog_bar=True)
        self.log('batch_acc', batch_acc, prog_bar=True)

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
        """Extract bio and batch embeddings."""
        self.eval()
        bio_z_list, batch_z_list = [], []

        with torch.no_grad():
            for batch in dataloader:
                data = batch[0].to(self.device)
                out = self(data)
                bio_z_list.append(out.bio_z.cpu())
                batch_z_list.append(out.batch_z.cpu())

        return torch.cat(bio_z_list).numpy(), torch.cat(batch_z_list).numpy()


class IMCModule(BaseBBNModule):
    """Lightning module for IMC data with MSE reconstruction."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model = IMCVAE(config.model)
        self.recon_loss = nn.MSELoss()

    def training_step(self, batch, batch_idx):
        data, batch_id, *_ = batch
        out = self(data)

        recon = self.recon_loss(data, out.reconstruction)
        kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
        kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()
        disc = self.ce_loss(out.bio_batch_pred, batch_id)
        clf = self.ce_loss(out.batch_batch_pred, batch_id)
        ortho = orthogonal_loss(out.bio_z, out.batch_z)

        loss = (
            self.lw.recon * recon +
            self.lw.discriminator * disc +
            self.lw.classifier * clf +
            self.lw.kl_bio * kl_bio +
            self.lw.kl_batch * kl_batch +
            self.lw.ortho * ortho
        )

        bio_acc = (out.bio_batch_pred.argmax(1) == batch_id).float().mean()
        batch_acc = (out.batch_batch_pred.argmax(1) == batch_id).float().mean()

        self._log_metrics(loss, recon, disc, clf, kl_bio, kl_batch, ortho, bio_acc, batch_acc)
        return loss


class RNAModule(BaseBBNModule):
    """Lightning module for scRNA-seq data with ZINB reconstruction."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model = GeneVAE(config.model)
        self.recon_loss = ZINBLoss()

    def training_step(self, batch, batch_idx):
        data, batch_id, *_ = batch
        out = self(data)

        recon = self.recon_loss(data, out.mean, out.disp, out.pi)
        kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
        kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()
        disc = self.ce_loss(out.bio_batch_pred, batch_id)
        clf = self.ce_loss(out.batch_batch_pred, batch_id)
        ortho = orthogonal_loss(out.bio_z, out.batch_z)
        kl_size = kl_divergence(out.size_mu, out.size_logvar).mean()

        loss = (
            self.lw.recon * recon +
            self.lw.discriminator * disc +
            self.lw.classifier * clf +
            self.lw.kl_bio * kl_bio +
            self.lw.kl_batch * kl_batch +
            self.lw.ortho * ortho +
            self.lw.kl_size * kl_size
        )

        bio_acc = (out.bio_batch_pred.argmax(1) == batch_id).float().mean()
        batch_acc = (out.batch_batch_pred.argmax(1) == batch_id).float().mean()

        self._log_metrics(loss, recon, disc, clf, kl_bio, kl_batch, ortho, bio_acc, batch_acc)
        self.log('kl_size', kl_size)
        return loss
