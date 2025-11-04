"""Training configuration for Stack-RNN experiments."""
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainingConfig:
    lr: float = 0.1
    lr_decay_factor: float = 0.5
    lr_min: float = 1e-5
    reg: float = 0.0
    reg_increase_factor: float = 2.0
    
    nepoch: int = 100
    nseq: int = 2000
    nmax: int = 5
    nmin: int = 2
    
    use_curriculum: bool = True
    curriculum_start: int = 3 # 最初のnmax
    curriculum_increment: int = 1
    
    bptt: int = 50
    nreset: int = 1000
    
    nval: int = 1000
    val_nmax_min: int = 20
    val_nmin: int = 2
    
    patience_epochs: int = 0
    decay_after_epoch: Optional[int] = None #Start decaying learning rate after this epoch
    
    save: bool = True
    save_dir: str = "results"
    save_best_only: bool = True
    
    seed: int = 1
    
    device: str = "cpu"
    
    def get_curriculum_nmax(self, epoch):
        """
        Get curriculum nmax for given epoch.
        C++ implementation (train_toy.cpp line 276):
            nmax = max(min(e+3, nmaxmax), 3)
        """
        if not self.use_curriculum:
            return self.nmax
        return max(min(epoch + 3, self.nmax), 3)
    
    def get_val_nmax(self, train_nmax):
        """Get validation nmax (at least val_nmax_min)."""
        return max(train_nmax, self.val_nmax_min)
    
    def should_decay_lr(self, epoch):
        """Check if LR should be decayed at this epoch."""
        if self.decay_after_epoch is not None:
            return epoch > self.decay_after_epoch
        # Default: C++ behavior (after nmax // 2 epochs)
        return epoch > self.nmax // 2
    
    def __str__(self):
        """String representation of config."""
        return (
            f"TrainingConfig(\n"
            f"  lr={self.lr}, lr_decay={self.lr_decay_factor}, lr_min={self.lr_min}\n"
            f"  nepoch={self.nepoch}, nseq={self.nseq}, nmax={self.nmax}\n"
            f"  curriculum={self.use_curriculum}, start={self.curriculum_start}\n"
            f"  bptt={self.bptt}, nreset={self.nreset}\n"
            f"  seed={self.seed}, device={self.device}\n"
            f")"
        )


def get_training_config(preset='default', **kwargs):
    """
    Get training configuration from preset.
    
    Args:
        preset (str): Preset name
        **kwargs: Override specific parameters
        
    Returns:
        TrainingConfig: Training configuration
    """

    config = TrainingConfig()

    if kwargs:
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    return config
