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

x = torch.randint(0, params['vocab_size'], (10, 32))
print('output encoder', x, "size", x.shape)
print(encoder.load_state_dict)
# hidden_state = encoder(x)
# print('output of encoder', hidden_state)