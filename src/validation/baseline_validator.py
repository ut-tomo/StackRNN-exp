import torch
import torch.nn as nn
from typing import Tuple

from src.data import generate_next_sequence, char_to_idx, set_seed

class BaselineValidator:
    def __init__(self, nval: int=1000, val_nmax_min: int=20, ishard: bool=False):
        """
        Args: 
            nval: バリデーションシーケンス数
            val_nmax_min: バリデーションのnmaxの最小値
            ishard: 未使用（StackRNN Validatorとのインターフェース互換性のため）
        """
        self.nval = nval
        self.val_nmax_min = val_nmax_min
        self.ishard = ishard
    
    def validate(
        self,
        model: torch.nn.Module,
        model_type: str,
        task_id: int,
        nchar: int,
        train_nmax: int,
        seed: int = 1,
    ) -> Tuple[float, float]:
        # [average_loss, accuracy]返す

        set_seed(seed)
        model.eval()
        
        val_nmax = max(train_nmax, self.val_nmax_min)
        
        total_loss = 0.0
        total_sequences_correct = 0
        total_sequences = 0
        
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        with torch.no_grad():
            for iseq in range(self.nval):
                sequence = generate_next_sequence(
                    val_nmax, 2, nchar, 1, ntask=task_id
                )
                indices = [char_to_idx(ch, nchar, task_id) for ch in sequence]
                
                if len(indices) < 2:
                    continue
                
                input_seq = torch.tensor(indices[:-1], dtype=torch.long)
                target_seq = torch.tensor(indices[1:], dtype=torch.long)
                
                if model_type == 'lstm':
                    output, _ = model(input_seq.unsqueeze(0))
                    output = output.squeeze(0)
                elif model_type in ['transformer', 'mamba']:
                    output = model(input_seq.unsqueeze(0))
                    output = output.squeeze(0)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Calculate loss for all positions
                loss_per_position = criterion(output, target_seq)
                total_loss += loss_per_position.sum().item()
                
                # Determine evaluation region based on task
                # C++ train_toy.cpp line 501-508
                eval_mask = torch.zeros(len(target_seq), dtype=torch.bool)
                is_eval = False
                
                for i in range(len(target_seq)):
                    cur = indices[i]  # Current character (input at position i)
                    next_char = indices[i + 1]  # Next character (target at position i)
                    
                    # Check if evaluation should start
                    if not is_eval:
                        if ((task_id == 1 and cur == 0 and next_char != 0) or
                            (task_id == 2 and cur == 0 and next_char != 0) or
                            (task_id == 3 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 4 and next_char == 0) or
                            (task_id == 5 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 6 and cur == 1 and next_char == 2) or
                            (task_id == 8 and i > 0 and sequence[i-1] == '=') or  # Task 8: after '=' marker
                            (task_id == 9 and i == len(sequence) - 1)):  # Task 9: only evaluate the last character (label)
                            is_eval = True
                    
                    eval_mask[i] = is_eval
                
                # Check if all evaluated positions are correct
                if eval_mask.any():
                    predictions = torch.argmax(output, dim=-1)
                    eval_correct = (predictions[eval_mask] == target_seq[eval_mask]).all().item()
                    if eval_correct:
                        total_sequences_correct += 1
                
                total_sequences += 1
        
        avg_loss = total_loss / (self.nval * val_nmax) if self.nval > 0 else 0.0
        avg_accuracy = total_sequences_correct / total_sequences if total_sequences > 0 else 0.0
        
        return avg_loss, avg_accuracy
