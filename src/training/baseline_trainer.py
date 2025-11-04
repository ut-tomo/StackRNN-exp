import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from src.data import generate_next_sequence, char_to_idx, set_seed
from src.config import TrainingConfig

class BaselineTrainer:
    def __init__(
        self, 
        model: nn.Module,
        model_type: str,
        task_id: int,
        nchar: int,
        config: TrainingConfig,
    ):
        self.model = model
        self.model_type = model_type
        self.task_id = task_id
        self.nchar = nchar
        self.config = config

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        # LR スケジューラ
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', #損失は小さいほどGOOD
            factor=config.lr_decay_factor, # 減衰率
            patience=5, 
            threshold=1e-4,
            min_lr=config.lr_min
        )
        self.criterion = nn.CrossEntropyLoss()
        
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        set_seed(config.seed)
        
        if config.save:
            os.makedirs(config.save_dir, exist_ok=True)
            self.log_file = os.path.join(config.save_dir, 'training.log')
            self._init_log()
            
    def _init_log(self):
        """Initialize training log file."""
        with open(self.log_file, 'w') as f:
            f.write(f"Baseline Model Training - {self.model_type.upper()}\n")
            f.write(f"Task: {self.task_id}, nchar: {self.nchar}\n")
            
            if hasattr(self.model, 'get_model_info'):
                info = self.model.get_model_info()
                f.write(f"Model info: {info}\n")
            else:
                total_params = sum(p.numel() for p in self.model.parameters())
                f.write(f"Model parameters: {total_params}\n")
            
            f.write(f"Configuration: {self.config}\n\n")
    
    def _log(self, message: str):
        """Log message to file and console."""
        print(message)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
            
    def train_epoch(self) -> float:
        self.model.train()
        current_nmax = self.config.get_curriculum_nmax(self.current_epoch)
        
        if self.current_epoch < 15:
            self._log(f"  [Curriculum] Epoch {self.current_epoch}: nmax = {current_nmax}")
        
        total_loss = 0.0
        total_chars = 0
        
        for iseq in range(self.config.nseq):
            sequence = generate_next_sequence(
                current_nmax, 2, self.nchar, nrep=1, ntask=self.task_id
            )
            indices = [char_to_idx(ch, self.nchar, self.task_id) for ch in sequence]
            
            if len(indices) < 2:
                continue
            
            input_seq = torch.tensor(indices[:-1], dtype=torch.long)
            target_seq = torch.tensor(indices[1:], dtype=torch.long)
            
            self.optimizer.zero_grad()
            
            if self.model_type == 'lstm':
                # LSTM expects batch dimension
                output, _ = self.model(input_seq.unsqueeze(0))
                output = output.squeeze(0)
            elif self.model_type in ['transformer', 'mamba']:
                # Transformer/Mamba batch forward
                output = self.model(input_seq.unsqueeze(0))
                output = output.squeeze(0)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
            loss = self.criterion(output, target_seq)

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()
            
            total_loss += loss.item() * (len(target_seq))
            total_chars += len(target_seq)
        
        avg_loss = total_loss / total_chars if total_chars > 0 else 0.0
        return avg_loss
    
    def train(self) -> Dict[str, Any]:
        self._log(f"\n{'='*70}")
        self._log(f"Starting training: {self.model_type.upper()} on Task {self.task_id}")
        self._log(f"{'='*70}\n")
        
        for epoch in range(self.config.nepoch):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss = self.train_epoch()
            
            # Always run validation
            val_loss, val_acc = self._validate()

            epoch_time = time.time() - start_time
            
            current_lr = self.optimizer.param_groups[0]['lr']
            log_str = (
                f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s | LR: {current_lr:.2e}")
            self._log(log_str)
            
            improved = self._check_improvement(epoch, val_loss, val_acc)
            
            self.scheduler.step(val_loss)
            
            #Early Stop
            if current_lr < self.config.lr_min:
                self._log(f"\n  → LR < {self.config.lr_min}, stopping training")
                break
            
            if not improved:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= 20:
                    self._log(f"\n  → No improvement for 20 epochs, stopping training")
                    break
        # Save final model
        if self.config.save and self.best_model_state is not None:
            self._save_best_model()
            
        self._log("\nTraining completed!")
    
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_model_state['val_acc'] if self.best_model_state else 0.0,
            'final_epoch': self.current_epoch,
            'final_lr': self.optimizer.param_groups[0]['lr'],
        }

    def _validate(self):
        """Validate using BaselineValidator."""
        from src.validation import BaselineValidator
        
        if not hasattr(self, '_baseline_validator'):
            self._baseline_validator = BaselineValidator(
                model_type=self.model_type,
                nval=1000,
                val_nmax_min=20
            )
        
        return self._baseline_validator.validate(
            self.model,
            self.task_id,
            self.nchar,
            self.config.get_curriculum_nmax(self.current_epoch),
            self.config.seed,
        )
    
    def _check_improvement(self, epoch: int, val_loss: float, val_acc: float) -> bool:
        """Check if validation improved and update best model."""
        if epoch == 0 or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            if self.config.save:
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }
            
            self.epochs_without_improvement = 0
            return True
        else:
            return False
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        model_info = self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        
        model_path = os.path.join(
            self.config.save_dir,
            f'model_{self.model_type}_ntask{self.task_id}_nchar{self.nchar}_seed{self.config.seed}.pt'
        )
        torch.save(self.best_model_state, model_path)
        
        self._log(f"\nBest model saved to {model_path}")
        self._log(f"Best Val Loss: {self.best_val_loss:.4f}, "
                 f"Best Val Acc: {self.best_model_state['val_acc']:.4f}")
