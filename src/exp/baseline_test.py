"""
Usage:
--model でlstm, transformer, mamba, allのいずれかを指定
--task で1から7までのタスクIDを指定
--seed でランダムシードを指定（デフォルト42）
--device でcpuかcudaを指定（デフォルトcpu）
"""

import torch
import argparse
from datetime import datetime

from src.models.baselines import LSTMModel, TransformerModel, MambaModel
from src.config import TrainingConfig
from src.training.baseline_trainer import BaselineTrainer


def get_task_config(task_id):
    """
    Get task-specific configuration.
    
    Task descriptions:
    1: a^nb^n (Counting)
    2: a^nb^kn (Counting with repetition)
    3: a^nb^mc^{n+m} (Addition)
    4: Reverse Copy
    5: a^nb^mc^{nm} (Multiplication)
    6: a^nb^mc^nd^m (Double counting)
    7: Binary Addition
    8: Reverse Polish Notation (RPN)
    9: Balanced Parentheses (Dyck Language)
    """
    task_configs = {
        1: {'nchar': 2, 'nmax': 50, 'curriculum_start': 3, 'description': 'a^nb^n'},
        2: {'nchar': 2, 'nmax': 50, 'curriculum_start': 3, 'description': 'a^nb^kn'},
        3: {'nchar': 3, 'nmax': 50, 'curriculum_start': 3, 'description': 'a^nb^mc^{n+m}'},
        4: {'nchar': 5, 'nmax': 30, 'curriculum_start': 3, 'description': 'Reverse Copy'},
        5: {'nchar': 3, 'nmax': 20, 'curriculum_start': 3, 'description': 'a^nb^mc^{nm}'},
        6: {'nchar': 4, 'nmax': 50, 'curriculum_start': 3, 'description': 'a^nb^mc^nd^m'},
        7: {'nchar': 5, 'nmax': 15, 'curriculum_start': 3, 'description': 'Binary Addition'},
        8: {'nchar': 4, 'nmax': 30, 'curriculum_start': 3, 'description': 'Reverse Polish Notation'},
        9: {'nchar': 3, 'nmax': 40, 'curriculum_start': 3, 'description': 'Balanced Parentheses (3 chars)'},
    }
    return task_configs.get(task_id, task_configs[1])


def get_model_config(model_type):
    configs = {
        'lstm': {
            'hidden_size': 10,
            'num_layers': 1,
            'dropout': 0.0,
        },
        'transformer': {
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'max_seq_len': 200,
        },
        'mamba': {
            'd_model': 64,
            'n_layers': 2,
            'd_state': 16,
            'expand_factor': 2,
            'd_conv': 4,
        }
    }
    return configs.get(model_type, configs['lstm'])


def create_model(model_type, nchar, model_config):
    if model_type == 'lstm':
        model = LSTMModel(
            nchar=nchar,
            nhid=model_config['hidden_size'],
            nlayers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.0)
        )
    elif model_type == 'transformer':
        model = TransformerModel(
            nchar=nchar,
            nhid=model_config['d_model'],
            nhead=model_config['nhead'],
            nlayers=model_config['num_layers'],
            dropout=model_config.get('dropout', 0.1),
            max_len=model_config.get('max_seq_len', 200)
        )
    elif model_type == 'mamba':
        model = MambaModel(
            nchar=nchar,
            nhid=model_config['d_model'],
            nlayers=model_config['n_layers'],
            d_state=model_config.get('d_state', 16),
            d_conv=model_config.get('d_conv', 4),
            expand_factor=model_config.get('expand_factor', 2)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_model(model_type, task_id, seed=42, device='cpu'):
    task_cfg = get_task_config(task_id)
    nchar = task_cfg['nchar']
    nmax = task_cfg['nmax']
    
    print(f"\n{'='*80}")
    print(f"Training {model_type.upper()} on Task {task_id}: {task_cfg['description']}")
    print(f"nchar={nchar}, nmax={nmax}, seed={seed}")
    print(f"{'='*80}\n")
    
    config = TrainingConfig(
        lr=0.1,
        lr_decay_factor=0.5,
        lr_min=1e-5,
        reg=0.0,
        
        nepoch=100,
        nseq=2000,
        nmax=nmax,
        nmin=2,
        
        use_curriculum=True,
        curriculum_start=task_cfg['curriculum_start'],
        curriculum_increment=1,
        
        bptt=50,
        nreset=1000,
        
        nval=1000,
        val_nmax_min=20,
        val_nmin=2,
        
        patience_epochs=0,
        decay_after_epoch=None,
        
        save=True,
        save_dir=f'results/baseline_{model_type}_task{task_id}_seed{seed}',
        save_best_only=True,
        
        seed=seed,
        device=device,
    )
    
    model_config = get_model_config(model_type)
    model = create_model(model_type, nchar, model_config)
    
    if device == 'cuda' and torch.cuda.is_available():
        model = model.to(device)
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    trainer = BaselineTrainer(
        model=model,
        model_type=model_type,
        task_id=task_id,
        nchar=nchar,
        config=config
    )
    
    results = trainer.train()
    
    print(f"\n{'='*80}")
    print(f"Training completed for {model_type.upper()} on Task {task_id}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final epoch: {results['final_epoch']}")
    print(f"Final learning rate: {results['final_lr']:.2e}")
    print(f"{'='*80}\n")
    
    return results


def run_all_experiments(models=None, tasks=None, seeds=None, device='cpu'):
    """
    Run experiments for all combinations of models, tasks, and seeds.
    
    Args:
        models: List of model types to train (default: ['lstm', 'transformer', 'mamba'])
        tasks: List of task IDs to train on (default: [1, 2, 3, 4, 5, 6, 7])
        seeds: List of random seeds (default: [42])
        device: Device to use ('cpu' or 'cuda')
    """
    if models is None:
        models = ['lstm', 'transformer', 'mamba']
    if tasks is None:
        tasks = [1, 2, 3, 4, 5, 6, 7]
    if seeds is None:
        seeds = [42]
    
    all_results = {}
    
    start_time = datetime.now()
    print(f"\n{'#'*80}")
    print(f"Starting baseline experiments")
    print(f"Models: {models}")
    print(f"Tasks: {tasks}")
    print(f"Seeds: {seeds}")
    print(f"Device: {device}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*80}\n")
    
    for model_type in models:
        for task_id in tasks:
            for seed in seeds:
                key = f"{model_type}_task{task_id}_seed{seed}"
                try:
                    results = train_model(model_type, task_id, seed, device)
                    all_results[key] = results
                except Exception as e:
                    print(f"\nERROR training {key}: {str(e)}")
                    all_results[key] = {'error': str(e)}
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'#'*80}")
    print(f"All experiments completed!")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"{'#'*80}\n")
    
    print("\nResults Summary:")
    print(f"{'Model':<15} {'Task':<6} {'Seed':<6} {'Best Val Loss':<15} {'Status'}")
    print("-" * 80)
    for key, result in all_results.items():
        parts = key.split('_')
        model = parts[0]
        task = parts[1]
        seed = parts[2]
        if 'error' in result:
            print(f"{model:<15} {task:<6} {seed:<6} {'ERROR':<15} {result['error'][:30]}")
        else:
            val_loss = result.get('best_val_loss', float('inf'))
            print(f"{model:<15} {task:<6} {seed:<6} {val_loss:<15.4f} {'SUCCESS'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Train baseline models on sequence tasks')
    parser.add_argument('--model', type=str, default='all',
                        choices=['all', 'lstm', 'transformer', 'mamba'],
                        help='Model type to train (default: all)')
    parser.add_argument('--task', type=int, default=None,
                        choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                        help='Task ID to train on (default: all tasks)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to use (default: cpu)')
    parser.add_argument('--all', action='store_true',
                        help='Run all combinations of models and tasks')
    
    args = parser.parse_args()
    
    if args.model == 'all':
        models = ['lstm', 'transformer', 'mamba']
    else:
        models = [args.model]
    
    if args.task is not None:
        tasks = [args.task]
    else:
        tasks = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    seeds = [args.seed]
    
    if args.all:
        run_all_experiments(models, tasks, seeds, args.device)
    else:
        if args.task is None:
            print("Error: Please specify --task or use --all flag")
            return
        train_model(models[0], tasks[0], seeds[0], args.device)


if __name__ == '__main__':
    main()
