import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model)) 
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x) 
    

class TransformerModel(nn.Module):
    def __init__(self, nchar, nhid, nhead=8, nlayers=6, dropout=0.1, max_len=1000):
        super(TransformerModel, self).__init__()
        self.nchar = nchar
        self.nhid = nhid
        self.nhead = nhead
        self.max_len = max_len
        
        self.embedding = nn.Embedding(nchar, nhid)
        self.pos_encoder = PositionalEncoding(nhid, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=nhid,
            nhead=nhead,
            dim_feedforward=nhid * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, nlayers)
        
        self.output_proj = nn.Linear(nhid, nchar)
        self.d_model = nhid
        self.init_weights()
        
    def init_weights(self):
        """
        U(-0.1, 0.1)による
        バイアスは0初期化
        """
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_proj.bias.data.zero_()
        self.output_proj.weight.data.uniform_(-initrange, initrange)
        
    def generate_square_subsequent_mask(self, sz):
        """
        [sz(seq length), sz]の行列で, 未来のトークンを-infでマスクしたテンソル
        
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k) + mask) V
        という形で使う
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, input_seq, mask=None, use_causal_mask=True):
        if input_seq.dim == 1:
            input_seq = input_seq.unsqueeze(0)  # バッチ次元を追加
            single_sequence = True
        else:
            single_sequence = False
        
        batch_size, seq_len = input_seq.size()
        device = input_seq.device
        
        embedded = self.embedding(input_seq) * math.sqrt(self.d_model)
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, nhid)
        embedded = self.pos_encoder(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, seq_len, nhid)
        
        if use_causal_mask:
            if mask is None:
                mask = self.generate_square_subsequent_mask(seq_len).to(device)
                
        transformer_out = self.transformer(embedded, mask=mask)
        output = self.output_proj(transformer_out)
        
        if single_sequence:
            output = output.squeeze(0)  # バッチ次元を削除
        
        return output
    
    def forward_step(self, input_seq, log_probs=False):
        """
        1ステップの予測を行う
        input_seq: (batch_size, seq_len)
        return: (batch_size, nchar)
        """
        if input_seq.dim() == 0:
            input_seq = input_seq.unsqueeze(0)
            
        full_output = self.forward(input_seq, use_causal_mask=True)
        output = full_output[-1]
        
        if log_probs:
            output = F.log_softmax(output, dim=-1)
            
        return output
    
    def init_hidden(self, batch_size):
        return None
    
    def get_model_info(self):
        """Return model information for logging"""
        return {
            'model_type': 'Transformer',
            'nchar': self.nchar,
            'nhid': self.nhid,
            'nhead': self.nhead,
            'nlayers': self.nlayers,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 