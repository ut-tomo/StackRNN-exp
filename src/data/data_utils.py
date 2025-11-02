import random
import numpy as np
import torch

def char_to_idx(ch, nchar, ntask=1):
    if ch == '':
        return 0

    if ntask == 7:
        if ch.isdigit():
            return int(ch)
        elif ch == '+':
            return 2
        elif ch == '=':
            return 3
        elif ch == '.':
            return 4
        else:
            raise ValueError(f"Invalid character for task 7: {ch}")
        
    idx = ord(ch) - ord('a') 
    return min(max(idx, 0), nchar - 1)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False