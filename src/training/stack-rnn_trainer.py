"""学習ロジック"""
import os
import time
import torch 
import torch.nn as nn
from typing import Any, Dict, Optional

from src.data import generate_next_sequence, char_to_idx, set_seed
from src.config import TrainingConfig


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        task_id: int,
        nchar: int,
        config: TrainingConfig,
        validator: Optional[Any] = None,
    ):
        self.model = model
        self.task_id = task_id
        self.nchar = nchar
        self.config = config
        self.validator = validator
        
        self.current_epoch = 0
        self.lr = config.lr
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.epochs_without_improvement = 0
        
        set_seed(config.seed)   
        
        if config.save:
            os.makedirs(config.save_dir, exist_ok=True)
            self.log_file = os.path.join(config.save_dir, 'training.log')
            self._init_log()
    
    def _init_log(self):
        with open(self.log_file, 'w') as f:
            f.write("Stack-RNN Training\n")
            f.write(f"Task: {self.task_id}, nchar: {self.nchar}\n")
            total_params = sum(p.numel() for p in self.model.parameters())
            f.write(f"Model parameters: {total_params}\n")
            f.write(f"Configuration: {self.config}\n\n")
            
    def _log(self, message: str):
        print(message)
        if hasattr(self, 'log_file'):
            with open(self.log_file, 'a') as f:
                f.write(message + '\n')
        
    def train_epoch(self) -> float:
        """
        一文字あたりの平均損失を返す
        """
        self.model.train()
        self.model.empty_stacks()
        cur = self.nchar - 1
        
        current_nmax = self.config.get_curriculum_nmax(self.current_epoch)
        
        if self.current_epoch < 15:
            self._log(f"  [Curriculum] Epoch {self.current_epoch}: nmax = {current_nmax}")
        
        total_loss = 0.0
        total_chars = 0
        
        for iseq in range(self.config.nseq):
            sequence = generate_next_sequence(
                current_nmax, 2, self.nchar, 1, self.task_id
            )
            
            # C++での Conditional Stack Reset
            # if(nreset == 1 || (nreset > 0 && iseq % nreset == 0))
            # 3つのモードがある
            # nreset == 1: 毎回リセット
            # nreset > 1: nresetごとにリセット
            # nreset == 0: リセットしない
            if self.config.nreset > 0 and iseq % self.config.nreset == 0 and iseq > 0:
                self.model.empty_stacks()
            
            for pos, ch in enumerate(sequence):
                next_char = char_to_idx(ch, self.nchar, self.task_id)
                
                output_probs, loss = self.model.forward_step(cur, next_char, training=True)
                
                #開始マーカーは勾配更新をスキップ, 代わりにempty_stacks()を呼ぶ
                if pos == 0 and iseq == 0:
                    self.model.empty_stacks()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=15.0)
                    
                    # Manual SGD update (no optimizer needed)
                    with torch.no_grad():
                        for param in self.model.parameters():
                            if param.grad is not None:
                                param.data -= self.lr * param.grad
                                param.grad.zero_()
                    
                    # BPTT: Detach hidden states periodically
                    if pos % self.config.bptt == 0:
                        self.model.detach_hidden_states()
                
                total_loss += loss.item()
                total_chars += 1
                
                cur = next_char
        
        avg_loss = total_loss / total_chars
        return avg_loss
                    
    def train(self) -> Dict[str, Any]:
        self._log(f"\n{'='*70}")
        self._log(f"Starting training: Task {self.task_id}")
        self._log(f"{'='*70}\n")
        
        for epoch in range(self.config.nepoch):
            self.current_epoch = epoch
            start_time = time.time()
            
            train_loss = self.train_epoch()
            
            if self.validator is not None:
                val_loss, val_acc = self._validate(
                    self.model,
                    self.task_id,
                    self.nchar,
                    self.config.get_val_nmax(epoch),
                    self.config.seed,
                )
            else:
                val_loss, val_acc = 0.0, 0.0
                
            epoch_time = time.time() - start_time

            log_str = (
                f"Epoch {epoch:3d} |  Train loss = {train_loss:.4f} | "
                f"Val loss = {val_loss:.4f} | Val acc = {val_acc:.4f} | "
                f"Time: {epoch_time:.2f}s | LR: {self.lr:.2e}"
            )
            self._log(log_str)
            improved = self._check_improvement(epoch, val_loss, val_acc)
            
            if not improved and self.config.should_decay_lr(epoch):
                self._decay_lr()
                
            if self.lr < self.config.lr_min:
                self._log(f"\n  → LR < {self.config.lr_min}, stopping training")
                break
        if self.config.save and self.best_model_state is not None:
            self._save_best_model()
        
        self._log("\nTraining completed!")
        
        return {
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_model_state['val_acc'] if self.best_model_state else 0.0,
            'final_epoch': self.current_epoch,
            'final_lr': self.lr,
        }
    
    def _check_improvement(self, epoch: int, val_loss: float, val_acc: float) -> bool:
        if epoch == 0 or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            
            if self.config.save:
                self.best_model_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'lr': self.lr,
                }
            
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
        
    def _decay_lr(self):
        self.lr *= self.config.lr_decay_factor
        self._log(f"  → LR decayed to {self.lr:.2e}")
        
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state['model_state_dict'])
            self._log(f"  → Restored best model from epoch {self.best_model_state['epoch']}")
        
        self.model.reg *= self.config.reg_increase_factor
        if self.model.reg > 0:
            self._log(f"  → Regularization increased to {self.model.reg:.2e}")
    
    def _save_best_model(self):
        """Save best model checkpoint."""
        model_path = os.path.join(
            self.config.save_dir,
            f'model_ntask{self.task_id}_nchar{self.nchar}_'
            f'nhid{self.model.n_hidden}_nstack{self.model.n_stack}_seed{self.config.seed}.pt'
        )
        torch.save(self.best_model_state, model_path)
        
        self._log(f"\nBest model saved to {model_path}")
        self._log(f"Best Val Loss: {self.best_val_loss:.4f}, "
                 f"Best Val Acc: {self.best_model_state['val_acc']:.4f}")