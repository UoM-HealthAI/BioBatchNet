import torch.nn as nn
import numpy as np
import torch
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import scanpy as sc
import pandas as pd
from .util import visualization
from .loss import kl_divergence, orthogonal_loss, ZINBLoss, MMDLoss
from .util import MetricTracker
from .evaluation import evaluate_nn


class EarlyStopping:
    def __init__(self, patience=10, delta_ratio=0.000):
        self.patience = patience
        self.delta_ratio = delta_ratio
        self.best_loss = 1e10
        self.counter = 0
        self.early_stop = False
        self.best_epoch = 0  

    def step(self, current_loss, epoch):
        delta = self.best_loss * self.delta_ratio
        if current_loss < self.best_loss - delta:
            self.best_loss = current_loss
            self.best_epoch = epoch
            self.counter = 0
            return True  
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False  

class Trainer:
    def __init__(self, config, model, optimizer, train_dataloader, eval_dataloader, scheduler, device, seed):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.dataset_name = self.config['name']

        self.device = device
        self.scheduler = scheduler
        self.seed = seed
        self.eval_sampling_seed = self.config['eval_sampling_seed']

        self.cfg_trainer = self.config['trainer']
        self.epochs = self.cfg_trainer['epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.if_imc = self.cfg_trainer['if_imc']
        self.loss_weights = self.config['loss_weights']
        self.early_stopping = EarlyStopping(patience=self.cfg_trainer['early_stop'])

        self.sampling_fraction = self.cfg_trainer['sampling_fraction'][self.dataset_name]
        self.logger = self.config.get_logger('trainer', self.config['trainer']['verbosity'])
 
        self.mse_recon = nn.MSELoss()
        self.zinb_recon = ZINBLoss().cuda()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss()

        self.metric_tracker = MetricTracker(
            'total_loss', 'recon_loss', 'batch_loss_z1', 'batch_loss_z2',
            'mmd_loss_1', 'kl_loss_1', 'kl_loss_2', 'ortho_loss', 
        )

        # Base directory for saving all checkpoints and results
        self.base_checkpoint_dir = self.config.save_dir
        self.checkpoint_dir = None  # Will be updated for each seed training

    def train(self):
        """
        Main training loop that handles multiple seeds.
            For each seed:
            1. Creates a dedicated directory
            2. Runs training and evaluation
            3. Saves checkpoints and results
            4. umap visualization
            Finally combines results from all seeds.
        """
        # Create dedicated directory for this seed
        self.checkpoint_dir = self.base_checkpoint_dir / f'seed_{self.seed}'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
        # Training loop for current seed
        for epoch in tqdm(range(1, self.epochs + 1)):
            current_train_loss = self._train_epoch(epoch, mode='imc' if self.if_imc else 'rna')  
            improved = self.early_stopping.step(current_train_loss, epoch)
            
            if improved:
                self._save_best_checkpoint(epoch)
                self.logger.info(f"New best loss: {self.early_stopping.best_loss:.4f} at epoch {epoch}")

            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

            if epoch % 5 == 0:
                evaluation_results = self._evaluate_epoch(epoch, sampling_fraction=0.3*self.sampling_fraction)
                # Log evaluation results without wandb
                self.logger.info(f"epoch {epoch} evaluation results: {evaluation_results}")    
                    
            # if epoch % self.save_period == 0:
            #     self._save_checkpoint(epoch)

        # Load best model for final evaluation
        self._load_best_model()
        
        # Skip final evaluation for API usage
        if self.config.config.get('trainer', {}).get('skip_intermediate_eval', False):
            evaluation_results = {}
            self.logger.info(f"Skipping final evaluation for API usage")
        else:
            evaluation_results = self._evaluate_epoch(epoch, sampling_fraction=self.sampling_fraction)
            self.logger.info(f"Evaluation results after training with seed {self.seed}: {evaluation_results}")
            
            # Save evaluation results for current seed
            results_df = pd.DataFrame(evaluation_results)
            results_df.to_csv(self.checkpoint_dir / f'seed_{self.seed}_evaluation_result.csv', index=True)

         
        return evaluation_results
    
    def _train_epoch(self, epoch, mode):
        self.metric_tracker.reset()
        self.model.train()

        total_correct_z1 = 0
        total_correct_z2 = 0
        total_samples = 0

        for data, batch_id in self.train_dataloader:
            data, batch_id = data.to(self.device), batch_id.to(self.device)
            self.optimizer.zero_grad()

            # forward pass and reconstrution loss
            if mode == 'imc':
                bio_z, bio_mu, bio_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, reconstruction = self.model(data)
                recon_loss = self.mse_recon(data, reconstruction)
            else:
                bio_z, bio_mu, bio_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi, size_factor, size_mu, size_logvar = self.model(data)
                recon_loss = self.zinb_recon(data, _mean, _disp, _pi)
            
            # bio kl loss 
            kl_loss_1 = kl_divergence(bio_mu, bio_logvar).mean()
            bio_z_prior = torch.randn_like(bio_z, device=self.device)
            mmd_loss = self.mmd_loss(bio_z, bio_z_prior)

            # batch kl loss
            kl_loss_2 = kl_divergence(batch_mu, batch_logvar).mean()

            # library size kl loss
            kl_loss_size = kl_divergence(size_mu, size_logvar).mean() if mode == 'rna' else 0

            # discriminator loss            
            batch_loss_z1 = self.criterion_classification(bio_batch_pred, batch_id)

            # classifier loss
            batch_loss_z2 = self.criterion_classification(batch_batch_pred, batch_id)
            
            # Orthogonal loss
            ortho_loss_value = orthogonal_loss(bio_z, batch_z)

            # Total loss
            loss = (self.loss_weights['recon_loss'] * recon_loss +
                    self.loss_weights['discriminator'] * batch_loss_z1 +
                    self.loss_weights['classifier'] * batch_loss_z2 +
                    self.loss_weights['mmd_loss_1'] * mmd_loss +
                    self.loss_weights['kl_loss_1'] * kl_loss_1 +
                    self.loss_weights['kl_loss_2'] * kl_loss_2 +
                    self.loss_weights['ortho_loss'] * ortho_loss_value +
                    (self.loss_weights['kl_loss_size'] * kl_loss_size if mode == 'rna' else 0))

            loss.backward()
            self.optimizer.step()

            # Update losses
            losses = {
                'total_loss': loss.item(),
                'recon_loss': recon_loss.item(),
                'batch_loss_z1': batch_loss_z1.item(),
                'batch_loss_z2': batch_loss_z2.item(),
                'mmd_loss_1': mmd_loss.item(),
                'kl_loss_1': kl_loss_1.item(),
                'kl_loss_2': kl_loss_2.item(),
                'ortho_loss': ortho_loss_value.item(),
            }
            self.metric_tracker.update_batch(losses, count=data.size(0))

            # Accuracy calculation
            z1_pred = torch.argmax(bio_batch_pred, dim=1)
            z2_pred = torch.argmax(batch_batch_pred, dim=1)

            total_correct_z1 += (z1_pred == batch_id).sum().item()
            total_correct_z2 += (z2_pred == batch_id).sum().item()

            total_samples += batch_id.size(0)
        self.scheduler.step()

        # Avg accuracy for epoch
        z1_accuracy = total_correct_z1 / total_samples * 100
        z2_accuracy = total_correct_z2 / total_samples * 100

        # Log accuracies (without wandb)
        # Could save to file or just use logger

        self.logger.info(
            f"Epoch {epoch}: "
            f"Loss = {self.metric_tracker.avg('total_loss'):.2f}, "
            f"KL Loss = {self.metric_tracker.avg('kl_loss_1'):.2f}, "
            f"Z1 Accuracy = {z1_accuracy:.2f}, "
            f"Z2 Accuracy = {z2_accuracy:.2f}"
        )

        return self.metric_tracker.avg('total_loss')

    def _evaluate_epoch(self, epoch, sampling_fraction):
        # Skip evaluation for API usage
        if self.config.config.get('trainer', {}).get('skip_intermediate_eval', False):
            return {}
            
        with torch.no_grad():
            self.model.eval()

            all_data, all_bio_z, all_batch_z, all_batch_ids, all_cell_types = [], [], [], [], []

            for data, batch_id, cell_type in self.eval_dataloader:
                data, batch_id = data.to(self.device), batch_id.to(self.device)

                if self.if_imc:
                    bio_z, _, _, batch_z, *_ = self.model(data)
                else:
                    bio_z, _, _, batch_z, *_ = self.model(data)
                
                all_data.append(data.cpu().numpy())
                all_bio_z.append(bio_z.cpu().numpy())
                all_batch_z.append(batch_z.cpu().numpy())
                all_batch_ids.append(batch_id.cpu().numpy())
                all_cell_types.append(cell_type.cpu().numpy())

            # Convert lists to numpy arrays
            raw_data = np.concatenate(all_data, axis=0)
            bio_integrated_data = np.concatenate(all_bio_z, axis=0)
            batch_integrated_data = np.concatenate(all_batch_z, axis=0)
            batch_ids = np.concatenate(all_batch_ids, axis=0)
            cell_types = np.concatenate(all_cell_types, axis=0)

            # adata_unintegrated
            adata_unintegrated = sc.AnnData(raw_data)
            adata_unintegrated.obs['BATCH'] = pd.Categorical(batch_ids)
            adata_unintegrated.obs['celltype'] = pd.Categorical(cell_types)

            # adata_bio (bio_z - biological features)
            adata_bio = sc.AnnData(bio_integrated_data)
            adata_bio.obs['BATCH'] = pd.Categorical(batch_ids)
            adata_bio.obs['celltype'] = pd.Categorical(cell_types)
            adata_bio.obsm['X_biobatchnet'] = bio_integrated_data

            # adata_batch (batch_z - batch features)
            adata_batch = sc.AnnData(batch_integrated_data)
            adata_batch.obs['BATCH'] = pd.Categorical(batch_ids)
            adata_batch.obs['celltype'] = pd.Categorical(cell_types)
            adata_batch.obsm['X_batch'] = batch_integrated_data

            # visualization
            fig_save_dir = self.checkpoint_dir / 'fig'
            fig_save_dir.mkdir(parents=True, exist_ok=True)  
            
            # Visualize bio_z (biological space)
            visualization(fig_save_dir, adata_bio, 'X_biobatchnet', epoch)
            
            # Visualize batch_z (batch space)
            visualization(fig_save_dir, adata_batch, 'X_batch', epoch)
            
            self.logger.info(f"figure saved at {fig_save_dir}")

            adata_dict = {'Raw': adata_unintegrated, 'BioBatchNet': adata_bio}
            evaluation_results = evaluate_nn(adata_dict, fraction=sampling_fraction, seed=self.eval_sampling_seed)
            
            return evaluation_results
    
    def _save_checkpoint(self, epoch):
        """
        Save model checkpoint for current epoch in seed-specific directory.
        
        Args:
            epoch (int): Current epoch number
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'loss_weights': self.loss_weights
        }
        
        # Save checkpoint in seed-specific directory
        filename = str(self.checkpoint_dir / f'checkpoint-epoch{epoch}.pth')
        torch.save(state, filename)
        self.logger.info(f"Saving checkpoint: {filename} ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path, weights_only=False)
        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']

        # load architecture params from checkpoint.
        if checkpoint['config']['arch'] != self.config['arch']:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint['state_dict'])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint['config']['optimizer']['type'] != self.config['optimizer']['type']:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        self.logger.info("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _save_best_checkpoint(self, epoch):
        """
        Save the best model checkpoint.
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'loss_weights': self.loss_weights,
            'best_loss': self.early_stopping.best_loss,
            'best_epoch': self.early_stopping.best_epoch
        }
        
        best_filename = str(self.checkpoint_dir / 'best_model.pth')
        torch.save(state, best_filename)
        self.logger.info(f"Saved best model: {best_filename}")

    def _load_best_model(self):
        """
        Load the best model for final evaluation.
        """
        best_path = self.checkpoint_dir / 'best_model.pth'
        if best_path.exists():
            self.logger.info(f"Loading best model from {best_path}")
            checkpoint = torch.load(best_path, weights_only=False)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.logger.info(f"Loaded best model from epoch {checkpoint['best_epoch']} with loss: {checkpoint['best_loss']:.4f}")
        else:
            self.logger.warning("Best model not found, using current model for evaluation")

    @staticmethod
    def calculate_final_results(all_evaluation_results):
        """Calculate mean and variance for all metrics across different seeds.
        
        Args:
            all_evaluation_results: List of evaluation results from different seeds
            
        Returns:
            Dictionary containing mean and variance for each metric
        """
        method = list(all_evaluation_results[0].keys())[0]  
        metrics = all_evaluation_results[0][method].keys()  
        
        final_results = {method: {}}
        
        for metric in metrics:
            values = [result[method][metric] for result in all_evaluation_results]
            mean_value = np.mean(values)
            std_value = np.std(values)
            final_results[method][metric] = {
                'mean': mean_value,
                'std': std_value
            }
        
        return final_results

    