"""
StackRNN実験スクリプト
全タスク（1-9）でStackRNNを訓練
"""
import argparse
import os
import sys
import torch

# Import with correct filenames (using hyphens)
import importlib
from src.models.stack_rnn import StackRNN
from src.config import TrainingConfig

trainer_module = importlib.import_module('src.training.stack-rnn_trainer')
validator_module = importlib.import_module('src.validation.stack-rnn_validator')
Trainer = trainer_module.Trainer
Validator = validator_module.Validator


# タスク設定
TASK_CONFIGS = {
    1: {'name': 'a^nb^n', 'nchar': 2, 'description': 'Counting task'},
    2: {'name': 'a^nb^2n', 'nchar': 2, 'description': 'Counting with repetition'},
    3: {'name': 'a^nb^mc^{n+m}', 'nchar': 3, 'description': 'Addition'},
    4: {'name': 'Reverse', 'nchar': 3, 'description': 'Reverse copy'},
    5: {'name': 'a^nb^mc^{nm}', 'nchar': 3, 'description': 'Multiplication'},
    6: {'name': 'a^nb^mc^nd^m', 'nchar': 4, 'description': 'Interleaved counting'},
    7: {'name': 'Binary Add', 'nchar': 14, 'description': 'Binary addition'},
    8: {'name': 'RPN', 'nchar': 4, 'description': 'Reverse Polish Notation'},
    9: {'name': 'Parentheses', 'nchar': 3, 'description': 'Balanced parentheses (3 chars)'},
}


def create_model(task_id, nchar, nhid=40, nstack=10, stack_size=200, device='cpu'):
    """Create StackRNN model."""
    model = StackRNN(
        nchar=nchar,
        nhid=nhid,
        nstack=nstack,
        stack_size=stack_size,
        depth=2,
        mod=1,  # rec with stack
        use_noop=False,
        reg=0.0
    )
    model = model.to(device)
    return model


def get_training_config(task_id, save_dir=None):
    """Get training configuration for task."""
    config = TrainingConfig(
        lr=0.1,
        lr_decay_factor=0.5,
        lr_min=1e-5,
        reg=0.0,
        reg_increase_factor=2.0,
        nepoch=100,
        nseq=2000,
        nmax=5,
        nmin=2,
        use_curriculum=True,
        curriculum_start=3,
        curriculum_increment=1,
        bptt=50,
        nreset=1000,
        nval=1000,
        val_nmax_min=20,
        val_nmin=2,
        patience_epochs=0,
        decay_after_epoch=None,
        seed=1,
        save=save_dir is not None,
        save_dir=save_dir if save_dir else './results'
    )
    return config


def train_model(task_id, device='cpu', seed=1, save_results=True):
    """Train StackRNN on specified task."""
    
    task_config = TASK_CONFIGS[task_id]
    nchar = task_config['nchar']
    
    print(f"\n{'='*70}")
    print(f"Task {task_id}: {task_config['name']}")
    print(f"Description: {task_config['description']}")
    print(f"nchar: {nchar}")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_model(task_id, nchar, device=device)
    
    # Setup save directory
    save_dir = None
    if save_results:
        save_dir = f'./results/stackrnn/task{task_id}'
        os.makedirs(save_dir, exist_ok=True)
    
    # Get training config
    config = get_training_config(task_id, save_dir)
    config.seed = seed
    
    # Create validator
    validator = Validator(nval=config.nval, val_nmax_min=config.val_nmax_min, ishard=False)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        task_id=task_id,
        nchar=nchar,
        config=config,
        validator=validator
    )
    
    # Train
    results = trainer.train()
    
    print(f"\n{'='*70}")
    print(f"Training completed for Task {task_id}")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    print(f"Final epoch: {results['final_epoch']}")
    print(f"Final learning rate: {results['final_lr']:.2e}")
    print(f"{'='*70}\n")
    
    return results


def run_all_tasks(device='cpu', seed=1, save_results=True):
    """Run all tasks."""
    results = {}
    
    for task_id in sorted(TASK_CONFIGS.keys()):
        try:
            result = train_model(task_id, device=device, seed=seed, save_results=save_results)
            results[task_id] = result
        except Exception as e:
            print(f"\n✗ Error in Task {task_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[task_id] = {'error': str(e)}
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY OF ALL TASKS")
    print(f"{'='*70}")
    print(f"{'Task':<6} {'Name':<20} {'Best Val Loss':<15} {'Best Val Acc':<15}")
    print(f"{'-'*70}")
    
    for task_id in sorted(results.keys()):
        task_name = TASK_CONFIGS[task_id]['name']
        if 'error' in results[task_id]:
            print(f"{task_id:<6} {task_name:<20} {'ERROR':<15} {'ERROR':<15}")
        else:
            val_loss = results[task_id]['best_val_loss']
            val_acc = results[task_id]['best_val_acc']
            print(f"{task_id:<6} {task_name:<20} {val_loss:<15.4f} {val_acc:<15.4f}")
    
    print(f"{'='*70}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train StackRNN on sequence tasks')
    parser.add_argument('--task', type=int, default=None,
                        help='Task ID (1-9). If not specified, run all tasks.')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    save_results = not args.no_save
    
    if args.task is not None:
        # Run single task
        if args.task not in TASK_CONFIGS:
            print(f"Error: Task {args.task} not found. Available tasks: {list(TASK_CONFIGS.keys())}")
            sys.exit(1)
        
        train_model(args.task, device=args.device, seed=args.seed, save_results=save_results)
    else:
        # Run all tasks
        run_all_tasks(device=args.device, seed=args.seed, save_results=save_results)


if __name__ == '__main__':
    main()
