from encoder import EncoderStack
from utils import load_param
import torch

params = load_param('transformer_params.yml')
model = EncoderStack(vocab_size=params['vocab_size'],
                     d_model=params['d_model'],
                     nhead=params['nhead'],
                     num_layers=params['num_layers'])

x = torch.randint(0, params['vocab_size'], (10, 32))
print('output encoder', x, "size", x.shape)
print(model.load_state_dict)