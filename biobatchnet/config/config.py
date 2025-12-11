"""Configuration management using dataclasses."""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    in_sz: int = 2000
    out_sz: int = 2000
    latent_sz: int = 20
    num_batch: int = 1

    bio_encoder_layers: List[int] = field(default_factory=lambda: [500, 2000, 2000])
    batch_encoder_layers: List[int] = field(default_factory=lambda: [500])
    decoder_layers: List[int] = field(default_factory=lambda: [2000, 2000, 500])
    bio_classifier_layers: List[int] = field(default_factory=lambda: [500, 2000, 2000])
    batch_classifier_layers: List[int] = field(default_factory=lambda: [128])

    dropout: float = 0.3


@dataclass
class LossConfig:
    """Loss weights configuration."""
    recon: float = 10.0
    discriminator: float = 0.3
    classifier: float = 1.0
    kl_bio: float = 0.001
    kl_batch: float = 0.1
    ortho: float = 0.01
    kl_size: float = 0.002  # only for GeneVAE/scRNA


@dataclass
class TrainerConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Scheduler
    scheduler_step: int = 50
    scheduler_gamma: float = 0.1

    # Early stopping
    early_stop: int = 15

    # Logging
    save_dir: str = "./saved/"
    save_period: int = 20
    verbosity: int = 2

    # Evaluation
    sampling_fraction: float = 1.0
    eval_seed: int = 42


@dataclass
class Config:
    """Master configuration."""
    name: str = "experiment"
    mode: str = "imc"  # "imc" or "rna"
    seed: int = 42

    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        yaml_path = Path(yaml_path)

        with open(yaml_path, 'r') as f:
            yaml_dict = yaml.safe_load(f)

        # Handle base config inheritance
        if '_base_' in yaml_dict:
            base_path = yaml_path.parent / yaml_dict['_base_']
            with open(base_path, 'r') as f:
                base_dict = yaml.safe_load(f)
            yaml_dict = _deep_merge(base_dict, yaml_dict)
            del yaml_dict['_base_']

        return cls._from_dict(yaml_dict)

    @classmethod
    def _from_dict(cls, d: dict) -> 'Config':
        """Create Config from dictionary."""
        model_dict = d.get('model', {})
        loss_dict = d.get('loss', {})
        trainer_dict = d.get('trainer', {})

        return cls(
            name=d.get('name', 'experiment'),
            mode=d.get('mode', 'imc'),
            seed=d.get('seed', 42),
            model=ModelConfig(**model_dict) if model_dict else ModelConfig(),
            loss=LossConfig(**loss_dict) if loss_dict else LossConfig(),
            trainer=TrainerConfig(**trainer_dict) if trainer_dict else TrainerConfig(),
        )

    @classmethod
    def from_preset(cls, dataset: str) -> 'Config':
        """Load configuration from preset for a specific dataset.

        Args:
            dataset: Dataset name, e.g., 'damond', 'pancreas', 'mousebrain'
        """
        presets_path = Path(__file__).parent / 'presets.yaml'
        with open(presets_path, 'r') as f:
            presets = yaml.safe_load(f)

        # Find dataset in presets
        preset = None
        mode = None
        for m in ['imc', 'rna']:
            if dataset in presets.get(m, {}):
                preset = presets[m][dataset]
                mode = m
                break

        if preset is None:
            available = []
            for m in ['imc', 'rna']:
                available.extend(presets.get(m, {}).keys())
            raise ValueError(f"Dataset '{dataset}' not found. Available: {available}")

        model_dict = preset.get('model', {})
        loss_dict = preset.get('loss', {})

        return cls(
            name=dataset,
            mode=mode,
            model=ModelConfig(**{**ModelConfig().__dict__, **model_dict}),
            loss=LossConfig(**{**LossConfig().__dict__, **loss_dict}),
        )

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            'name': self.name,
            'mode': self.mode,
            'seed': self.seed,
            'model': self.model.__dict__,
            'loss': self.loss.__dict__,
            'trainer': self.trainer.__dict__,
        }
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
