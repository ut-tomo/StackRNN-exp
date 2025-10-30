import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MambaBlock(nn.Module):
    """Simplified Mamba/State Space Model block"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super(MambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand_factor * d_model
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # State space parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # For B and C
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # A parameter (state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Skip connection
        residual = x
        x = self.norm(x)
        
        # Input projection
        x_and_res = self.in_proj(x)  # (batch_size, seq_len, d_inner * 2)
        x, res = x_and_res.split(self.d_inner, dim=-1)
        
        # Apply SiLU activation to x
        x = F.silu(x)
        
        # 1D convolution for local context
        x = x.transpose(1, 2)  # (batch_size, d_inner, seq_len)
        x = self.conv1d(x)[:, :, :seq_len]  # (batch_size, d_inner, seq_len)
        x = x.transpose(1, 2)  # (batch_size, seq_len, d_inner)
        
        # Apply SiLU activation after conv
        x = F.silu(x)
        
        # State space computation
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Generate B and C from input
        x_proj = self.x_proj(x)  # (batch_size, seq_len, d_state * 2)
        B, C = x_proj.chunk(2, dim=-1)  # Each: (batch_size, seq_len, d_state)
        
        # Generate delta (time step)
        delta = F.softplus(self.dt_proj(x))  # (batch_size, seq_len, d_inner)
        
        # Simplified state space computation using scan
        y = self.selective_scan(x, delta, A, B, C)
        
        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x
        
        # Apply gate
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection
        return output + residual

    def selective_scan(self, x, delta, A, B, C):
        """
        Simplified selective scan operation
        Args:
            x: (batch_size, seq_len, d_inner)
            delta: (batch_size, seq_len, d_inner)
            A: (d_inner, d_state)
            B: (batch_size, seq_len, d_state)
            C: (batch_size, seq_len, d_state)
        Returns:
            y: (batch_size, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        ys = []
        for i in range(seq_len):
            # Current inputs
            x_i = x[:, i]  # (batch_size, d_inner)
            delta_i = delta[:, i]  # (batch_size, d_inner)
            B_i = B[:, i]  # (batch_size, d_state)
            C_i = C[:, i]  # (batch_size, d_state)
            
            # Discretize A and B
            delta_A = torch.exp(delta_i.unsqueeze(-1) * A.unsqueeze(0))  # (batch_size, d_inner, d_state)
            delta_B = delta_i.unsqueeze(-1) * B_i.unsqueeze(1)  # (batch_size, d_inner, d_state)
            
            # State update
            h = h * delta_A + x_i.unsqueeze(-1) * delta_B
            
            # Output
            y_i = torch.sum(h * C_i.unsqueeze(1), dim=-1)  # (batch_size, d_inner)
            ys.append(y_i)
        
        return torch.stack(ys, dim=1)  # (batch_size, seq_len, d_inner)

class MambaModel(nn.Module):
    def __init__(self, nchar, nhid, nlayers=6, d_state=16, d_conv=4, expand_factor=2):
        super(MambaModel, self).__init__()
        self.nchar = nchar
        self.nhid = nhid
        self.nlayers = nlayers
        
        # Embedding layer
        self.embedding = nn.Embedding(nchar, nhid)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(nhid, d_state, d_conv, expand_factor)
            for _ in range(nlayers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(nhid)
        
        # Output projection
        self.output_proj = nn.Linear(nhid, nchar)
        
        # Initialize parameters
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_seq):
        """
        Args:
            input_seq: (batch_size, seq_len) or (seq_len,) for single sequence
        Returns:
            output: (batch_size, seq_len, nchar) or (seq_len, nchar)
        """
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension
            single_sequence = True
        else:
            single_sequence = False
            
        # Embedding
        x = self.embedding(input_seq)  # (batch_size, seq_len, nhid)
        
        # Apply Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Output projection
        output = self.output_proj(x)  # (batch_size, seq_len, nchar)
        
        if single_sequence:
            output = output.squeeze(0)  # Remove batch dimension
            
        return output

    def forward_step(self, input_seq, log_probs=False):
        """
        Forward step for autoregressive generation
        Args:
            input_seq: (seq_len,) input sequence up to current position
            log_probs: if True, return log probabilities
        Returns:
            output: (nchar,) output probabilities for next token
        """
        if input_seq.dim() == 0:
            input_seq = input_seq.unsqueeze(0)
            
        # Get full output
        full_output = self.forward(input_seq)  # (seq_len, nchar)
        
        # Return last position output
        output = full_output[-1]  # (nchar,)
        
        if log_probs:
            output = F.log_softmax(output, dim=-1)
            
        return output

    def init_hidden(self, batch_size):
        return None

    def get_model_info(self):
        """Return model information for logging"""
        return {
            'model_type': 'Mamba',
            'nchar': self.nchar,
            'nhid': self.nhid,
            'nlayers': self.nlayers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }