from dataclasses import dataclass, field
from typing import Dict, List
import yaml
from pathlib import Path


@dataclass
class MethodConfig:
    embed: str
    need_pca: bool
    type: str  # 'nn' or 'non_nn'
    preprocess: List[str]  # ['hvg', 'normalize', 'log1p', 'pca']


@dataclass
class DatasetConfig:
    path: str
    mode: str  # 'imc' or 'seq'
    sampling_fraction: float
    sampling_seed: int
    seed_list: List[int]


@dataclass
class BaselineConfig:
    methods: Dict[str, MethodConfig] = field(default_factory=dict)
    datasets: Dict[str, DatasetConfig] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> 'BaselineConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        methods = {}
        for name, cfg in data.get('methods', {}).items():
            methods[name] = MethodConfig(
                embed=cfg['embed'],
                need_pca=cfg['need_pca'],
                type=cfg['type'],
                preprocess=cfg['preprocess']
            )

        datasets = {}
        for name, cfg in data.get('datasets', {}).items():
            datasets[name] = DatasetConfig(
                path=cfg['path'],
                mode=cfg['mode'],
                sampling_fraction=cfg['sampling_fraction'],
                sampling_seed=cfg['sampling_seed'],
                seed_list=cfg['seed_list']
            )

        return cls(methods=methods, datasets=datasets)

    def get_nn_methods(self) -> Dict[str, MethodConfig]:
        """Return methods of type 'nn'."""
        return {k: v for k, v in self.methods.items() if v.type == 'nn'}

    def get_non_nn_methods(self) -> Dict[str, MethodConfig]:
        """Return methods of type 'non_nn'."""
        return {k: v for k, v in self.methods.items() if v.type == 'non_nn'}

    def get_method(self, name: str) -> MethodConfig:
        """Get a specific method config by name."""
        return self.methods[name]

    def get_dataset(self, name: str) -> DatasetConfig:
        """Get a specific dataset config by name."""
        return self.datasets[name]
