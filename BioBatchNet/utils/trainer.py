import torch.nn as nn
import numpy as np
import torch
import wandb
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import random
import scanpy as sc
import pandas as pd

from Baseline.evaluation import evaluate_NN
from .loss import kl_divergence, orthogonal_loss, ZINBLoss, MMDLoss
from .util import MetricTracker

class Trainer:
    def __init__(self, config, model, optimizer, train_dataloader, eval_dataloader, scheduler, device):
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        self.device = device
        self.scheduler = scheduler
        self.train_seed_list = self.config['train_seed_list']
        self.eval_sampling_seed = self.config['eval_sampling_seed']

        self.cfg_trainer = self.config['trainer']
        self.epochs = self.cfg_trainer['epochs']
        self.save_period = self.cfg_trainer['save_period']
        self.if_imc = self.cfg_trainer['if_imc']
        self.loss_weights = self.cfg_trainer['loss_weights'] 

        self.checkpoint_dir = self.config.save_dir
        self.logger = self.config.get_logger('trainer', self.config['trainer']['verbosity'])
 
        self.mse_recon = nn.MSELoss()
        self.zinb_recon = ZINBLoss().cuda()
        self.criterion_classification = nn.CrossEntropyLoss()
        self.mmd_loss = MMDLoss()

        wandb.init(project=config['name'], config=config)

        self.metric_tracker = MetricTracker(
            'total_loss', 'recon_loss', 'kl_loss_1', 'kl_loss_2', 
            'ortho_loss', 'batch_loss_z1', 'batch_loss_z2'
        )

    def train(self):
        all_evaluation_results = []  # Used to store all evaluation results
        seed_list = self.config['train_seed_list']  # Read the seed list from the config

        for seed in seed_list:
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            for epoch in tqdm(range(1, self.epochs + 1)):
                self._train_epoch(epoch, mode='imc' if self.if_imc else 'rna')            
                if epoch % self.save_period == 0:
                    self._save_checkpoint(epoch)

            # Evaluate the model after training using the eval_sampling_seed
            evaluation_results = self._evaluate_model()
            all_evaluation_results.append(evaluation_results)  

            self.logger.info(f"Evaluation results after training with seed {seed}: {evaluation_results}")

        # Calculate the mean and variance of all evaluation results
        final_results = self.calculate_final_results(all_evaluation_results)
        self.logger.info(f"Final evaluation results: {final_results}")
        final_results_df = pd.DataFrame(final_results)
        final_results_df.to_csv(self.config.save_dir / 'final_results.csv', index=False)

    def _train_epoch(self, epoch, mode):
        self.metric_tracker.reset()
        self.model.train()

        total_correct_z1 = 0
        total_correct_z2 = 0
        total_samples = 0

        for data, batch_id, _ in self.dataloader:
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
                    self.loss_weights['kl_loss_1'] * mmd_loss +
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

        # log to wandb
        self.metric_tracker.log_to_wandb({
            'Z1 Accuracy': z1_accuracy,
            'Z2 Accuracy': z2_accuracy
        })

        self.logger.info(
            f"Epoch {epoch}: "
            f"Loss = {self.metric_tracker.avg('total_loss'):.2f}, "
            f"KL Loss = {self.metric_tracker.avg('kl_loss_1'):.2f}, "
            f"Z1 Accuracy = {z1_accuracy:.2f}, "
            f"Z2 Accuracy = {z2_accuracy:.2f}"
        )

    def _evaluate_model(self):
        with torch.no_grad():
            self.model.eval()

            all_data, all_bio_z, all_batch_ids, all_cell_types = [], [], [], []

            for data, batch_id, cell_type in self.eval_dataloader:
                data, batch_id = data.to(self.device), batch_id.to(self.device)

                bio_z, *_ = self.model(data)
                
                all_data.append(data.cpu().numpy())
                all_bio_z.append(bio_z.cpu().numpy())
                all_batch_ids.append(batch_id.cpu().numpy())
                all_cell_types.append(cell_type.cpu().numpy())

            # Convert lists to numpy arrays
            raw_data = np.concatenate(all_data, axis=0)
            integrated_data = np.concatenate(all_bio_z, axis=0)
            batch_ids = np.concatenate(all_batch_ids, axis=0)
            cell_types = np.concatenate(all_cell_types, axis=0)

            # adata_unintegrated
            adata_unintegrated = sc.AnnData(raw_data)
            adata_unintegrated.obs['batch_id'] = batch_ids
            adata_unintegrated.obs['cell_type'] = cell_types

            # adata_post (integrated data)
            adata_post = sc.AnnData(raw_data)
            adata_post.obs['batch_id'] = batch_ids
            adata_post.obs['cell_type'] = cell_types
            adata_post.obs['X_biobatchnet'] = integrated_data

            adata_dict = {'Raw': adata_unintegrated, 'BioBatchNet': adata_post}
            evaluation_results = evaluate_NN(adata_dict, seed=self.eval_sampling_seed)
    
            return evaluation_results
    
    def _save_checkpoint(self, epoch):
        """
        Saving checkpoints

        :param epoch: current epoch number
        :param log: logging information of the epoch
        :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
        """
        arch = type(self.model).__name__
        state = {
            'arch': arch,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'loss_weights': self.loss_weights  # Save the loss weights
        }
        filename = str(self.checkpoint_dir / 'checkpoint-epoch{}.pth'.format(epoch))
        torch.save(state, filename)
        self.logger.info("Saving checkpoint: {} ...".format(filename))

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info("Loading checkpoint: {} ...".format(resume_path))
        checkpoint = torch.load(resume_path)
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

    def calculate_final_results(self, all_evaluation_results):
        # Assuming all_evaluation_results is a list of dictionaries with the same keys
        metrics = all_evaluation_results[0].keys()
        final_results = {}

        for metric in metrics:
            # Collect all values for this metric
            values = [result[metric] for result in all_evaluation_results]
            # Calculate mean and variance
            mean_value = np.mean(values)
            variance_value = np.var(values)
            # Store in final results
            final_results[metric] = {'mean': mean_value, 'variance': variance_value}

        return final_results

    