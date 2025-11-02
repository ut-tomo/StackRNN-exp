import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

from src.data import generate_next_sequence, char_to_idx, set_seed
from src.config import TrainingConfig