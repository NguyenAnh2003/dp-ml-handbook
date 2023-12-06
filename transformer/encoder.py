import torch
import torch.nn as nn
from position_encoding import PositionEncoding
from torch.nn.functional import gelu
class EncoderStack(nn.Module):
    """
        position embedding - raw
        input embedding - how
        encoder - principle
        """
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        """
        Embedding: 1. num_embeddings
                   2. embedding_dim -> 300?
                   4. padding_idx
                   5. device
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) # ??
        self.position_encoding = PositionEncoding(d_model=d_model)
        self.dropout = nn.Dropout(p=0.1)
        # encoder_layer, num_layers, mask check
        self.encoder = nn.TransformerEncoder(
            # Encoder layer d_model, nhead, dropout
            nn.TransformerEncoderLayer(d_model=d_model,
                                       nhead=nhead, activation="gelu"),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.position_encoding(x)
        x = self.dropout(x)
        output = self.encoder(x)
        return output