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
    
    