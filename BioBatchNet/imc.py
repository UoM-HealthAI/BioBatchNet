import argparse
import collections
import torch
import os
import numpy as np
import random
import pandas as pd

from parse_config import ConfigParser
import models.model as model
from utils.dataset import IMCDataset
from utils.util import set_random_seed
from utils.trainer import Trainer


def main(config):
    logger = config.get_logger('train')

    dataset_name = config['name']
    dataset = IMCDataset(dataset_name)
    train_dataloader = config.init_obj('train_dataloader', torch.utils.data , dataset)
    eval_dataloader = config.init_obj('eval_dataloader', torch.utils.data , dataset)
    device = 'cuda' 

    all_evaluation_results = []
    for seed in config['train_seed_list']:
        set_random_seed(seed)
        BioBatchNet = config.init_obj('arch', model)
        logger.info(BioBatchNet)
        BioBatchNet = BioBatchNet.to(device)

        # optimizer - simplified with unified learning rate
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
        
        # Start training
        logger.info("------------------training begin------------------")
        result_df = trainer.train()
        all_evaluation_results.append(result_df)
    
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
    
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--data', '--data_name'], type=str, target='data_loader;type') 
    ]
    config = ConfigParser.from_args(args, options)
    main(config)


