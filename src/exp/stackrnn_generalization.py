"""
StackRNN Generalization Experiment
Train on short sequences (nmax=20), test on longer sequences (up to nmax=59)

Based on train_toy.cpp line 450-520:
- Tests sequence lengths from 2 to 59 (1 increment)
- Uses 200 test sequences per length

Usage:
    # Default: test on all lengths 2-59 
    python3 -m src.exp.stackrnn_generalization --task 8 --train-nmax 20
    
    # Custom test lengths
    python3 -m src.exp.stackrnn_generalization --task 8 --train-nmax 20 --test-nmax 10 20 30 40 50 60
"""
import argparse
import os
import sys
import torch
import json
from datetime import datetime

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
    9: {'name': 'Parentheses', 'nchar': 4, 'description': 'Balanced parentheses classification'},
}


def create_model(task_id, nchar, nhid=40, nstack=10, stack_size=200, device='cpu'):
    """Create StackRNN model."""
    model = StackRNN(
        nchar=nchar,
        nhid=nhid,
        nstack=nstack,
        stack_size=stack_size,
        depth=2,
        mod=1,
        use_noop=False,
        reg=0.0
    )
    model = model.to(device)
    return model


def get_training_config(train_nmax=20, save_dir=None, seed=1):
    """Get training configuration."""
    config = TrainingConfig(
        lr=0.1,
        lr_decay_factor=0.5,
        lr_min=1e-5,
        reg=0.0,
        reg_increase_factor=2.0,
        nepoch=100,
        nseq=2000,
        nmax=train_nmax,
        nmin=2,
        use_curriculum=True,
        curriculum_start=3,
        curriculum_increment=1,
        bptt=50,
        nreset=1000,
        nval=200,  # 各系列長で200シーケンス（論文に合わせる）
        val_nmax_min=20,
        val_nmin=2,
        patience_epochs=0,
        decay_after_epoch=None,
        seed=seed,
        save=save_dir is not None,
        save_dir=save_dir if save_dir else './results'
    )
    return config


def evaluate_generalization(model, task_id, nchar, test_nmax_values, seed=1):
    """
    Evaluate model on different sequence lengths.
    
    Args:
        model: Trained model
        task_id: Task ID
        nchar: Number of characters
        test_nmax_values: List of nmax values to test
        seed: Random seed
    
    Returns:
        Dictionary mapping nmax to (loss, accuracy)
    """
    results = {}
    validator = Validator(nval=200, val_nmax_min=20, ishard=False)  # 各長さで200シーケンス
    
    for test_nmax in test_nmax_values:
        print(f"  Evaluating on nmax={test_nmax}...", end=' ')
        val_loss, val_acc = validator.validate(
            model=model,
            task_id=task_id,
            nchar=nchar,
            train_nmax=test_nmax,
            seed=seed
        )
        results[test_nmax] = {'loss': val_loss, 'accuracy': val_acc}
        print(f"Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    return results


def run_generalization_experiment(
    task_id, 
    train_nmax=20, 
    test_nmax_values=None,
    device='cpu', 
    seed=1, 
    save_results=True
):
    """
    Run generalization experiment: train on train_nmax, test on test_nmax_values.
    """
    
    if test_nmax_values is None:
        # 論文に合わせて2から59まで1刻み（C++ train_toy.cpp line 450参照）
        test_nmax_values = list(range(2, 60))
    
    task_config = TASK_CONFIGS[task_id]
    nchar = task_config['nchar']
    
    print(f"\n{'='*70}")
    print(f"Generalization Experiment: Task {task_id} - {task_config['name']}")
    print(f"{'='*70}")
    print(f"Description: {task_config['description']}")
    print(f"nchar: {nchar}")
    print(f"Train nmax: {train_nmax}")
    print(f"Test nmax values: {test_nmax_values}")
    print(f"{'='*70}\n")
    
    # Create model
    model = create_model(task_id, nchar, device=device)
    
    # Setup save directory
    save_dir = None
    if save_results:
        save_dir = f'./results/stackrnn_generalization/task{task_id}_trainmax{train_nmax}'
        os.makedirs(save_dir, exist_ok=True)
    
    # Get training config
    config = get_training_config(train_nmax=train_nmax, save_dir=save_dir, seed=seed)
    
    # Create validator for training
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
    print(f"Training on sequences up to length {train_nmax}...")
    train_results = trainer.train()
    
    print(f"\n{'='*70}")
    print(f"Training completed")
    print(f"Best validation loss: {train_results['best_val_loss']:.4f}")
    print(f"Best validation accuracy: {train_results['best_val_acc']:.4f}")
    print(f"Final epoch: {train_results['final_epoch']}")
    print(f"{'='*70}\n")
    
    # Evaluate generalization
    print(f"Evaluating generalization on different sequence lengths...")
    gen_results = evaluate_generalization(
        model, task_id, nchar, test_nmax_values, seed=seed
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"Generalization Results Summary")
    print(f"{'='*70}")
    print(f"{'nmax':<10} {'Loss':<15} {'Accuracy':<15}")
    print(f"{'-'*40}")
    for nmax in sorted(gen_results.keys()):
        loss = gen_results[nmax]['loss']
        acc = gen_results[nmax]['accuracy']
        marker = " *" if nmax == train_nmax else ""
        print(f"{nmax:<10} {loss:<15.4f} {acc:<15.4f}{marker}")
    print(f"{'='*70}")
    print("* = training length")
    
    # Save results
    if save_results and save_dir:
        results_data = {
            'task_id': task_id,
            'task_name': task_config['name'],
            'train_nmax': train_nmax,
            'test_nmax_values': test_nmax_values,
            'training': {
                'best_val_loss': train_results['best_val_loss'],
                'best_val_acc': train_results['best_val_acc'],
                'final_epoch': train_results['final_epoch'],
            },
            'generalization': gen_results,
            'timestamp': datetime.now().isoformat(),
            'seed': seed,
        }
        
        results_file = os.path.join(save_dir, 'generalization_results.json')
        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    return train_results, gen_results


def main():
    parser = argparse.ArgumentParser(
        description='StackRNN Generalization Experiment: Train on short sequences, test on longer ones'
    )
    parser.add_argument('--task', type=int, required=True,
                        choices=list(TASK_CONFIGS.keys()),
                        help='Task ID (1-9)')
    parser.add_argument('--train-nmax', type=int, default=20,
                        help='Maximum sequence length for training (default: 20)')
    parser.add_argument('--test-nmax', type=int, nargs='+', default=None,
                        help='Maximum sequence lengths for testing (default: 2-59, matching paper)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cpu or cuda)')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    
    args = parser.parse_args()
    
    save_results = not args.no_save
    
    run_generalization_experiment(
        task_id=args.task,
        train_nmax=args.train_nmax,
        test_nmax_values=args.test_nmax,
        device=args.device,
        seed=args.seed,
        save_results=save_results
    )


if __name__ == '__main__':
    main()
