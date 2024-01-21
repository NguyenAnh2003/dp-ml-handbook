from encoder import EncoderStack
from utils import load_param
import torch

params = load_param('transformer_params.yml')
print(params)
encoder = EncoderStack(vocab_size=params['vocab_size'],
                     d_model=params['d_model'],
                     nhead=params['nhead'],
                     num_layers=params['num_layers'],
                     max_seq_len=params['max_len'])

# batch size 30, seq len 10, passed through embedding -> 30x10x768
x = torch.randn(1, 12, params['d_model'])
print('input encoder', x, "size", x.shape)
print(encoder.load_state_dict)
# hidden_state = encoder(x)
# print('output of encoder', hidden_state)