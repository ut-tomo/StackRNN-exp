import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, nchar, nhid, nlayers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.nchar = nchar
        self.nhid = nhid
        self.nlayers = nlayers

        self.embedding = nn.Embedding(nchar, nhid)
        self.lstm = nn.LSTM(nhid, nhid, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)

        self.output_proj = nn.Linear(nhid, nchar)

        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)
        
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)  # forget gate bias
                
    def init_hidden(self, batch_size):
        device = next(self.parameters()).device
        h0 = torch.zeros(self.nlayers, batch_size, self.nhid, device=device)
        c0 = torch.zeros(self.nlayers, batch_size, self.nhid, device=device)
        return (h0, c0)
    
    def forward(self, input_seq, hidden=None):
        """
        Args:
            input_seq: Tensor of shape (batch_size, seq_length) containing input character indices.
            hidden: Tuple of (h0, c0) for LSTM initial hidden and cell states.
        """
        if input_seq.dim() == 1:
            input_seq = input_seq.unsqueeze(0)  # Add batch dimension if missing
            single_sequence = True
        else:
            single_sequence = False
        
        batch_size, seq_len = input_seq.size()
        
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        embedded = self.embedding(input_seq)  # (batch_size, seq_length, nhid)
        lstm_out, hidden = self.lstm(embedded, hidden)  # lstm_out: (batch_size, seq_length, nhid)
        output = self.output_proj(lstm_out)  # (batch_size, seq_length, nchar)
        
        if single_sequence:
            output = output.squeeze(0)  # Remove batch dimension if it was added
        
        return output, hidden
    
    def forward_step(self, input_char, hidden, log_probs=False):
        """
        Args:
            input_char: Tensor of shape (batch_size,) containing input character indices.
            hidden: Tuple of (h, c) for LSTM hidden and cell states.
        Returns:
            output: Tensor of shape (batch_size, nchar) containing output logits for the next character.
            hidden: Updated hidden and cell states tuple (h, c).
        """
        if isinstance(input_char, int):
            input_char = torch.tensor([input_char], device=next(self.parameters()).device)
        elif input_char.dim() == 0:
            input_char = input_char.unsqueeze(0) # Add batch dimension if missing

        single_input = input_char.size(0) == 1 # Check if single input
        
        input_seq = input_char.unsqueeze(1)  # (batch_size, 1)
        
        output, hidden = self.forward(input_seq, hidden)
        output = output.squeeze(1)  # (batch_size, nchar)
        
        if log_probs:
            output = F.log_softmax(output, dim=-1)
        
        if single_input and output.dim() > 1:
            output = output.squeeze(0)
            
        return output, hidden

    def get_model_info(self):
        """Return model information for logging"""
        return {
            'model_type': 'LSTM',
            'nchar': self.nchar,
            'nhid': self.nhid,
            'nlayers': self.nlayers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }