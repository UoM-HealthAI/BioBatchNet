import optuna
import pandas as pd
import torch
import yaml
from pathlib import Path
import numpy as np
import sys
import os

# Add BioBatchNet directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'BioBatchNet'))

from scrna import main
from parse_config import ConfigParser
import logging

# Set logging level to reduce output
logging.getLogger('optuna').setLevel(logging.WARNING)

def objective(trial):
    """
    Objective function for Optuna optimization
    """
    # Suggest hyperparameter ranges
    recon_loss = trial.suggest_float('recon_loss', 0.5, 10.0, log=True)
    kl_loss_1 = trial.suggest_float('kl_loss_1', 1e-7, 1e-4, log=True)
    discriminator = trial.suggest_float('discriminator', 0.005, 0.2, log=True)
    
    print(f"Trying params: recon={recon_loss:.4f}, kl1={kl_loss_1:.2e}, disc={discriminator:.4f}")
    
    # Load configuration
    base_config_path = 'BioBatchNet/config/scRNA/pancreas.yaml'
    config = load_and_modify_config(base_config_path, {
        'recon_loss': recon_loss,
        'kl_loss_1': kl_loss_1,
        'discriminator': discriminator
    })
    
    try:
        # Run quick experiment
        result = run_quick_experiment(config)
        
        # Calculate objective score (ARI + NMI)
        ari = result['BioBatchNet']['ARI']['mean']
        nmi = result['BioBatchNet']['NMI']['mean']
        combined_score = ari + nmi
        
        print(f"Results: ARI={ari:.4f}, NMI={nmi:.4f}, Combined={combined_score:.4f}")
        
        # Record intermediate results
        trial.set_user_attr('ARI', ari)
        trial.set_user_attr('NMI', nmi)
        trial.set_user_attr('ARI_std', result['BioBatchNet']['ARI']['std'])
        trial.set_user_attr('NMI_std', result['BioBatchNet']['NMI']['std'])
        
        return combined_score
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        return 0.0  # Return worst score

def deep_merge(base_dict, update_dict):
    """
    Deep merge two dictionaries
    """
    result = base_dict.copy()
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

def load_and_modify_config(config_path, modifications):
    """Load configuration file and modify parameters"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base configuration inheritance
    if '_base_' in config:
        base_config_path = Path(config_path).parent / config['_base_']
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        # Deep merge configurations
        config_without_base = {k: v for k, v in config.items() if k != '_base_'}
        config = deep_merge(base_config, config_without_base)
    
    # Modify loss weights
    if 'loss_weights' not in config:
        config['loss_weights'] = {}
    
    config['loss_weights']['recon_loss'] = modifications['recon_loss']
    config['loss_weights']['kl_loss_1'] = modifications['kl_loss_1'] 
    config['loss_weights']['discriminator'] = modifications['discriminator']
    
    return config

def run_quick_experiment(config):
    """Run quick experiment with reduced training time"""
    # Quick search configuration
    config['train_seed_list'] = [42]  # Use only 1 seed
    config['trainer']['epochs'] = 100  # Reduce epochs
    config['trainer']['early_stop'] = 10  # Early stopping
    config['trainer']['skip_intermediate_eval'] = True  # Skip intermediate evaluation
    
    # Create temporary config parser
    config_parser = ConfigParser(config)
    
    # Run main function
    main(config_parser)
    
    # Read results
    result_path = config_parser.save_dir / 'final_results.csv'
    if result_path.exists():
        result_df = pd.read_csv(result_path, index_col=0)
        return result_df.to_dict()
    else:
        raise Exception("Result file not found")

def run_optuna_search(n_trials=20):
    """
    Run Optuna hyperparameter search
    """
    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner()
    )
    
    print(f"Starting Optuna search with {n_trials} trials...")
    
    # Optimize
    study.optimize(objective, n_trials=n_trials)
    
    # Output best results
    print("\n=== Best Results ===")
    best_trial = study.best_trial
    print(f"Best Combined Score: {best_trial.value:.4f}")
    print(f"Best parameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    print(f"Best performance:")
    print(f"  ARI: {best_trial.user_attrs['ARI']:.4f} ± {best_trial.user_attrs['ARI_std']:.4f}")
    print(f"  NMI: {best_trial.user_attrs['NMI']:.4f} ± {best_trial.user_attrs['NMI_std']:.4f}")
    
    # Save all results
    trials_df = study.trials_dataframe()
    trials_df.to_csv('optuna_search_results.csv', index=False)
    
    # Save best parameter configuration
    best_config = {
        'loss_weights': {
            'recon_loss': best_trial.params['recon_loss'],
            'kl_loss_1': best_trial.params['kl_loss_1'],
            'discriminator': best_trial.params['discriminator']
        }
    }
    
    with open('best_hyperparameters.yaml', 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    return study

if __name__ == "__main__":
    # Run search
    study = run_optuna_search(n_trials=30)  # 30 trials
    
    # Visualize results (optional)
    try:
        import matplotlib.pyplot as plt
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig('optimization_history.png')
        
        optuna.visualization.matplotlib.plot_param_importances(study)
        plt.savefig('param_importances.png')
        print("Visualization results saved as optimization_history.png and param_importances.png")
    except ImportError:
        print("Install matplotlib to generate visualization results") 