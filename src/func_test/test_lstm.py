import pytest
import torch
from models.baselines.lstm_model import LSTMModel

class TestLSTM:
    @pytest.fixture
    def model(self):
        return LSTMModel(nchar=10,  nhid=32, nlayers=2)
    def test_initialization(self, model):
        assert model.nchar == 10
        assert model.nhid == 32
        assert model.nlayers == 2
        
    def test_forward_shape(self, model):
        input_seq = torch.randint(0, 10, (2, 5))
        output, hidden = model(input_seq)
        assert output.shape == (2, 5, 10)
    
    def test_forward_step_shape(self, model):
        hidden = model.init_hidden(2)
        input_char = torch.randint(0, 10, (2,))
        output, hidden = model.forward_step(input_char, hidden)
        assert output.shape == (2, 10)