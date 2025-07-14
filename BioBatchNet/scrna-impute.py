import argparse
import collections
import torch
import pandas as pd
import numpy as np

from parse_config import ConfigParser
import models.model as model
from utils.dataset import GeneDataset
from utils.util import prepare_device, set_random_seed
from utils.trainer import Trainer

def main(config):
    logger = config.get_logger('train')
    
    # prepare data
    dataset_name = config['name']       
    dataset = GeneDataset(dataset_name)
    target_data, masked_input, n, row, col = dataset.simulate_dropout(in_place=True)
    train_dataloader = config.init_obj('train_dataloader', torch.utils.data , dataset)
    eval_dataloader = config.init_obj('eval_dataloader', torch.utils.data , dataset)
    device, _ = prepare_device(config['n_gpu'])

    # if we want to also train scvi with the masked data add:
    # scvi.model.SCVI.setup_anndata(dataset.adata, batch_key="BATCH")
    # model = scvi.model.SCVI(dataset.adata, ...., gene_likelihood='zinb')

    all_evaluation_results = []
    for seed in config['train_seed_list']:
        set_random_seed(seed)
        BioBatchNet = config.init_obj('arch', model)
        logger.info(BioBatchNet)
        BioBatchNet = BioBatchNet.to(device)

        # Initialize optimizer and lr_scheduler
        trainable_params = filter(lambda p: p.requires_grad, BioBatchNet.parameters())
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)
        
        # trainer
        trainer = Trainer(config, 
                        model = BioBatchNet, 
                        optimizer = optimizer, 
                        train_dataloader = train_dataloader,
                        eval_dataloader = eval_dataloader,
                        scheduler = lr_scheduler, 
                        device = device,
                        seed = seed)
        
        logger.info("------------------training begin------------------")
        result_df = trainer.train()
        all_evaluation_results.append(result_df)
        
        print("\nIMPUTATION RESULT: ")
        input = torch.tensor(dataset.data, dtype=torch.float32)
        with torch.no_grad():
            bio_z, bio_mu, bio_logvar, batch_z, batch_mu, batch_logvar, bio_batch_pred, batch_batch_pred, _mean, _disp, _pi, size_factor, size_mu, size_logvar = BioBatchNet(input)
        
        imputed_means = _mean[row, col]
        imputed_pis = _pi[row,col]

        reconstructed = (1 - imputed_pis) * imputed_means
        error = np.abs(target_data-reconstructed.detach().numpy())
        mae = np.mean(error)
        print(f"Masked MAE: {mae:.4f}")
        

    
    base_checkpoint_dir = config.save_dir
    final_results = trainer.calculate_final_results(all_evaluation_results)
    final_results_df = pd.DataFrame(final_results)
    final_results_df.to_csv(base_checkpoint_dir / 'final_results.csv', index=True)
    logger.info("All experiments completed.")


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='BioBatchNet training')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--data', '--data_name'], type=str, target='data_loader;type') 
    ]
    config = ConfigParser.from_args(args, options)
    main(config)

    