import torch 
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, nchar, nhid, nlayers=1, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.nchar = nchar
        self.nhid = nhid
        self.nlayers = nlayers

        self.embedding == nn.Embedding(nchar, nhid)
        self.lstm = nn.LSTM(nhid, nhid, nlayers, batch_first=True, dropout=dropout if nlayers > 1 else 0)
        
        self.out_proj = nn.Linear(nhid, nchar)
        
        self.init_weights()