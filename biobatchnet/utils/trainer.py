"""Trainer for batch effect correction models."""
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import scanpy as sc
import pandas as pd

from .util import visualization, MetricTracker
from .loss import kl_divergence, orthogonal_loss, ZINBLoss
from .evaluation import evaluate_nn
from ..config import Config


class EarlyStopping:
    """Early stopping handler."""
    def __init__(self, patience=10):
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0

    def step(self, loss, epoch):
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


class Trainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloader,
        eval_dataloader=None,
        device='cuda',
    ):
        self.config = config
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader or train_dataloader
        self.device = device

        # Training config
        tc = config.trainer
        self.epochs = tc.epochs
        self.early_stopping = EarlyStopping(patience=tc.early_stop)

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=tc.scheduler_step,
            gamma=tc.scheduler_gamma,
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.zinb_loss = ZINBLoss().to(device)
        self.ce_loss = nn.CrossEntropyLoss()

        # Loss weights
        self.loss_weights = config.loss

        # Checkpointing
        self.save_dir = Path(tc.save_dir) / config.name / f'seed_{config.seed}'
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.logger = logging.getLogger('trainer')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)

        # Mode
        self.mode = config.mode  # 'imc' or 'rna'

        # Metrics
        self.metric_tracker = MetricTracker(
            'total_loss', 'recon_loss', 'disc_loss', 'clf_loss',
            'kl_bio', 'kl_batch', 'ortho_loss',
        )

    def train(self):
        """Main training loop."""
        for epoch in tqdm(range(1, self.epochs + 1)):
            loss = self._train_epoch(epoch)
            improved = self.early_stopping.step(loss, epoch)

            if improved:
                self._save_checkpoint(epoch, best=True)
                self.logger.info(f"Best loss: {loss:.4f} at epoch {epoch}")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        self._load_best_checkpoint()

    def _train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        self.metric_tracker.reset()

        total_correct_bio = 0
        total_correct_batch = 0
        total_samples = 0

        for data, batch_id in self.train_dataloader:
            data = data.to(self.device)
            batch_id = batch_id.to(self.device)
            self.optimizer.zero_grad()

            # Forward
            out = self.model(data)

            # Reconstruction loss
            if self.mode == 'imc':
                recon_loss = self.mse_loss(data, out.reconstruction)
            else:
                recon_loss = self.zinb_loss(data, out.mean, out.disp, out.pi)

            # KL losses
            kl_bio = kl_divergence(out.bio_mu, out.bio_logvar).mean()
            kl_batch = kl_divergence(out.batch_mu, out.batch_logvar).mean()
            kl_size = 0
            if self.mode == 'rna':
                kl_size = kl_divergence(out.size_mu, out.size_logvar).mean()

            # Classification losses
            disc_loss = self.ce_loss(out.bio_batch_pred, batch_id)
            clf_loss = self.ce_loss(out.batch_batch_pred, batch_id)

            # Orthogonal loss
            ortho_loss = orthogonal_loss(out.bio_z, out.batch_z)

            # Total loss
            lw = self.loss_weights
            loss = (
                lw.recon * recon_loss +
                lw.discriminator * disc_loss +
                lw.classifier * clf_loss +
                lw.kl_bio * kl_bio +
                lw.kl_batch * kl_batch +
                lw.ortho * ortho_loss +
                (lw.kl_size * kl_size if self.mode == 'rna' else 0)
            )

            loss.backward()
            self.optimizer.step()

            # Track metrics
            self.metric_tracker.update_batch({
                'total_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'disc_loss': disc_loss.item(),
                'clf_loss': clf_loss.item(),
                'kl_bio': kl_bio.item(),
                'kl_batch': kl_batch.item(),
                'ortho_loss': ortho_loss.item(),
            }, count=data.size(0))

            # Accuracy
            total_correct_bio += (out.bio_batch_pred.argmax(1) == batch_id).sum().item()
            total_correct_batch += (out.batch_batch_pred.argmax(1) == batch_id).sum().item()
            total_samples += batch_id.size(0)

        self.scheduler.step()

        avg_loss = self.metric_tracker.avg('total_loss')
        bio_acc = total_correct_bio / total_samples * 100
        batch_acc = total_correct_batch / total_samples * 100

        self.logger.info(
            f"Epoch {epoch}: Loss={avg_loss:.2f}, "
            f"BioAcc={bio_acc:.1f}%, BatchAcc={batch_acc:.1f}%"
        )

        return avg_loss

    def evaluate(self, sampling_fraction=1.0):
        """Evaluate model and return metrics."""
        self.model.eval()
        all_bio_z, all_batch_z, all_batch_ids, all_cell_types = [], [], [], []

        with torch.no_grad():
            for batch in self.eval_dataloader:
                if len(batch) == 3:
                    data, batch_id, cell_type = batch
                else:
                    data, batch_id = batch
                    cell_type = torch.zeros_like(batch_id)

                data = data.to(self.device)
                out = self.model(data)

                all_bio_z.append(out.bio_z.cpu().numpy())
                all_batch_z.append(out.batch_z.cpu().numpy())
                all_batch_ids.append(batch_id.numpy())
                all_cell_types.append(cell_type.numpy())

        bio_z = np.concatenate(all_bio_z)
        batch_z = np.concatenate(all_batch_z)
        batch_ids = np.concatenate(all_batch_ids)
        cell_types = np.concatenate(all_cell_types)

        # Create AnnData
        adata = sc.AnnData(bio_z)
        adata.obs['BATCH'] = pd.Categorical(batch_ids)
        adata.obs['celltype'] = pd.Categorical(cell_types)
        adata.obsm['X_biobatchnet'] = bio_z

        return evaluate_nn(
            {'biobatchnet': adata},
            fraction=sampling_fraction,
            seed=self.config.trainer.eval_seed,
        )

    def _save_checkpoint(self, epoch, best=False):
        """Save checkpoint."""
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.early_stopping.best_loss,
        }
        filename = 'best_model.pth' if best else f'checkpoint-{epoch}.pth'
        torch.save(state, self.save_dir / filename)

    def _load_best_checkpoint(self):
        """Load best checkpoint."""
        path = self.save_dir / 'best_model.pth'
        if path.exists():
            ckpt = torch.load(path, weights_only=False)
            self.model.load_state_dict(ckpt['model_state'])
            self.logger.info(f"Loaded best model from epoch {ckpt['epoch']}")
