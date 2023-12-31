import torch
import torch.nn as nn
from transformer_embedding import TransformerEmbedding
class EncoderStack(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len: int):
        super().__init__()
        self.embedding = TransformerEmbedding(vocab_size=vocab_size,
                                              d_model=d_model,
                                              max_len=max_seq_len,
                                              dropout=0.1,
                                              device='cpu')
        # encoder_layer, num_layers, mask check
        self.encoder = nn.TransformerEncoder(
            # Encoder layer d_model, nhead, dropout
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=nhead, activation="gelu"),
            num_layers=num_layers
        )

    def forward(self, x: torch.FloatTensor):
        # x: shape (seq_length, batch_size)
        x = self.embedding(x)
        output = self.encoder(x)
        return output