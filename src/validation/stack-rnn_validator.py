import torch
from typing import Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import generate_next_sequence, char_to_idx, set_seed


class Validator:
    """
    Implements C++-equivalent validation:
    - Continuous processing across sequences (no resets)
    - 1000 validation sequences
    - Minimum nmax of 20 for validation
    - Uses hard actions (argmax) during validation if ishard=True
    """
    
    def __init__(self, nval: int = 1000, val_nmax_min: int = 20, ishard: bool = False):
        self.nval = nval
        self.val_nmax_min = val_nmax_min
        self.ishard = ishard
    
    def validate(
        self,
        model: torch.nn.Module,
        task_id: int,
        nchar: int,
        train_nmax: int,
        seed: int = 1,
    ) -> Tuple[float, float]:

        set_seed(seed)
        
        model.eval()
        val_nmax = max(train_nmax, self.val_nmax_min)
        
        total_loss = 0.0
        total_correct = 0
        total_chars = 0
        
        # Per-sequence evaluation counters (for tasks that evaluate partial sequences)
        seq_correct = 0
        seq_total = 0
        total_sequences_correct = 0
        total_sequences = 0
        
        model.empty_stacks()
        
        cur = nchar - 1
        is_eval = False  # Flag to track if we're in evaluation mode for current sequence
        
        with torch.no_grad():
            for iseq in range(self.nval):
                sequence = generate_next_sequence(val_nmax, 2, nchar, 1, task_id)
                
                # Reset evaluation state at the beginning of each sequence
                is_eval = False
                seq_correct = 0
                seq_total = 0
                
                for ip, ch in enumerate(sequence):
                    next_char = char_to_idx(ch, nchar, task_id)
                    
                    output_probs, loss = model.forward_step(cur, next_char, is_hard=self.ishard, training=False)
                    
                    # Always compute loss
                    total_loss += loss.item()
                    total_chars += 1
                    
                    if not is_eval:
                        if ((task_id == 1 and cur == 0 and next_char != 0) or
                            (task_id == 2 and cur == 0 and next_char != 0) or
                            (task_id == 3 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 4 and next_char == 0) or
                            (task_id == 5 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 6 and cur == 1 and next_char == 2) or
                            (task_id == 9 and cur == 1 and next_char == 2)):
                            is_eval = True
                    
                    # Count accuracy only during evaluation period
                    if is_eval:
                        predicted = torch.argmax(output_probs)
                        if predicted.item() == next_char:
                            seq_correct += 1
                        seq_total += 1
                    
                    cur = next_char
                
                # At the end of each sequence, check if it was perfectly correct
                if seq_total > 0 and seq_correct == seq_total:
                    total_sequences_correct += 1
                total_sequences += 1
        
        avg_loss = total_loss / total_chars
        
        # Use sequence-level accuracy (percentage of perfectly predicted sequences)
        # This matches C++ behavior: ecorr / neval
        avg_accuracy = total_sequences_correct / total_sequences if total_sequences > 0 else 0.0
        
        return avg_loss, avg_accuracy
    
    def validate_exact(
        self,
        model: torch.nn.Module,
        task_id: int,
        nchar: int,
        nmax: int,
        seed: int = 1,
    ) -> Tuple[float, float]:

        return self.validate(model, task_id, nchar, nmax, seed)
