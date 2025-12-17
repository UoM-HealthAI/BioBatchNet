"""Unified training script for BioBatchNet."""
import argparse
import torch

from .config import Config
from .models.model import IMCVAE, GeneVAE
from .utils.dataset import BatchDataset
from .utils.trainer import Trainer
from .utils.util import set_random_seed


def train(config: Config, seed: int = 42):
    """Train model with given config and seed.

    Args:
        config: Configuration object
        seed: Random seed for reproducibility
    """
    set_random_seed(seed)
    config.seed = seed

    # Load dataset
    dataset = BatchDataset.from_preset(config.name)

    # Select model based on mode
    if config.mode == 'imc':
        model = IMCVAE(config.model)
    else:
        model = GeneVAE(config.model)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
    )

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train
    trainer = Trainer(
        config=config,
        model=model,
        train_dataloader=dataloader,
        eval_dataloader=dataloader,
        device=device,
    )
    trainer.train()

    return trainer


def main():
    parser = argparse.ArgumentParser(description='BioBatchNet Training')
    parser.add_argument('--preset', type=str, help='Dataset preset (damond, pancreas, etc.)')
    parser.add_argument('--config', type=str, help='Path to config yaml file')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    if args.preset:
        config = Config.from_preset(args.preset)
    elif args.config:
        config = Config.from_yaml(args.config)
    else:
        parser.error('Either --preset or --config is required')

    train(config, args.seed)


if __name__ == '__main__':
    main()
