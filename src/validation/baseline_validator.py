import torch
import torch.nn as nn
from typing import Tuple

from src.data import generate_next_sequence, char_to_idx, set_seed

class BaselineValidator:
    def __init__(self, model_type: str, nval: int=1000, val_nmax_min: int=20):
        """
        Args: 
            model_type: 'lstm', 'transformer', 'mamba' TODO: 他のモデルの追加?
            nval: バリデーションシーケンス数
            val_nmax_min: バリデーションのnmaxの最小値
        """
        self.model_type = model_type
        self.nval = nval
        self.val_nmax_min = val_nmax_min
    
    def validate(
        self,
        model: torch.nn.Module,
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
        total_correct = 0
        total_chars = 0
        
        criterion = nn.CrossEntropyLoss(reduction='sum')
        
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
                
                if self.model_type == 'lstm':
                    output, _ = model(input_seq.unsqueeze(0))
                    output = output.squeeze(0)
                elif self.model_type in ['transformer', 'mamba']:
                    output = model(input_seq.unsqueeze(0))
                    output = output.squeeze(0)
                else:
                    raise ValueError(f"Unknown model type: {self.model_type}")
                
                # Calculate loss
                loss = criterion(output, target_seq)
                total_loss += loss.item()
                
                # Calculate accuracy
                predictions = torch.argmax(output, dim=-1)
                correct = (predictions == target_seq).sum().item()
                total_correct += correct
                total_chars += len(target_seq)
        
        avg_loss = total_loss / total_chars if total_chars > 0 else 0.0
        avg_accuracy = total_correct / total_chars if total_chars > 0 else 0.0
        
        return avg_loss, avg_accuracy
