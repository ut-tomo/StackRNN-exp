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
    
    def __init__(self, nval: int = 1000, val_nmax_min: int = 20, ishard: bool = False, reset_per_sequence: bool = True):
        self.nval = nval
        self.val_nmax_min = val_nmax_min
        self.ishard = ishard
        self.reset_per_sequence = reset_per_sequence
    
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

                if self.reset_per_sequence:
                    model.empty_stacks()
            
                is_eval = False
                seq_correct = 0
                seq_total = 0
                
                for ip, ch in enumerate(sequence):
                    next_char = char_to_idx(ch, nchar, task_id)
                    
                    output_probs, loss = model.forward_step(cur, next_char, is_hard=self.ishard, training=False)
                
                    total_loss += loss.item()
                    total_chars += 1

                    # Task-specific evaluation period
                    if not is_eval:
                        if ((task_id == 1 and cur == 0 and next_char != 0) or
                            (task_id == 2 and cur == 0 and next_char != 0) or
                            (task_id == 3 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 4 and next_char == 0) or
                            (task_id == 5 and cur == nchar - 2 and next_char == nchar - 1) or
                            (task_id == 6 and cur == 1 and next_char == 2) or
                            (task_id == 8 and ip > 0 and sequence[ip-1] == '=') or
                            (task_id == 9 and ip == len(sequence) - 2)):
                            is_eval = True
                    
                    # Count accuracy during evaluation period
                    if is_eval:
                        predicted = torch.argmax(output_probs)
                        if predicted.item() == next_char:
                            seq_correct += 1
                        seq_total += 1
                        
                        # Task 9: only evaluate once (the label prediction)
                        if task_id == 9:
                            is_eval = False
                    
                    cur = next_char
                
                # check if it was perfectly correct at the end of each sequence
                if seq_total > 0:
                    if seq_correct == seq_total:
                        total_sequences_correct += 1
                    total_correct += seq_correct
                total_sequences += 1
        
        avg_loss = total_loss / total_chars
        avg_accuracy = total_correct / total_chars if total_chars > 0 else 0.0
        
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
    
    def seq_test(
        self,
        model: torch.nn.Module,
        task_id: int,
        nchar: int,
        nmax: int,
        ntest: int = 1000,
        seed: int = 1,
    ) -> Tuple[float, float]:
        """
        C++ test evaluation (lines 455-525 in train_toy.cpp)
        
        Key differences from validation:
        - Task 1-6: No stack reset between sequences (continuous)
        - Task 7+: Reset stacks before each sequence
        - Sequence-level accuracy only (perfect match required)
        - Task-specific evaluation periods only
        """
        set_seed(seed)
        model.eval()
        
        iscountfirstelement = (task_id != 4)
        
        total_loss = 0.0
        total_chars = 0

        nmin = nmax
        
        corr = 0  # Correct predictions in current evaluation period
        ecorr = 0  # Number of perfectly correct sequences
        sseq = 0  # Length of current evaluation period
        neval = 0  # Number of sequences evaluated
        
        cur = nchar - 1
        is_eval = False
        
        with torch.no_grad():
            for iseq in range(ntest):
                # Task 1-6: No reset (continuous processing)
                # Task 7+: Reset before each sequence
                if task_id >= 7:
                    model.empty_stacks()
            
                sequence = generate_next_sequence(nmax, nmin, nchar, 1, task_id)
                
                is_eval = False
            
                for ip, ch in enumerate(sequence):
                    next_char = char_to_idx(ch, nchar, task_id)
                
                    output_probs, loss = model.forward_step(cur, next_char, is_hard=True, training=False)
                
                    if ip == 0:
                        if iseq != 0:
                            neval += 1
                            predicted_first = torch.argmax(output_probs).item()
                            if corr == sseq and (not iscountfirstelement or predicted_first == next_char):
                                ecorr += 1
                        sseq = 0
                        corr = 0
                        is_eval = False
                
                    predicted = torch.argmax(output_probs).item()
                    if is_eval and predicted == next_char:
                        corr += 1
                    if is_eval:
                        sseq += 1
                
                    total_loss += loss.item()
                    total_chars += 1
                
                    if ((task_id == 1 and cur == 0 and next_char != 0) or
                        (task_id == 2 and cur == 0 and next_char != 0) or
                        (task_id == 3 and cur == nchar - 2 and next_char == nchar - 1) or
                        (task_id == 4 and next_char == 0) or
                        (task_id == 5 and cur == nchar - 2 and next_char == nchar - 1) or
                        (task_id == 6 and cur == 1 and next_char == 2)):
                        is_eval = True
                
                    cur = next_char
            
            if neval < ntest:
                neval += 1
                if corr == sseq:
                    ecorr += 1
        
        avg_loss = total_loss / total_chars if total_chars > 0 else 0.0
        avg_accuracy = ecorr / neval if neval > 0 else 0.0
        
        return avg_loss, avg_accuracy
