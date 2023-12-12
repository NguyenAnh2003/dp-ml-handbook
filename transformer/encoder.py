import torch
import torch.nn as nn
from position_encoding import PositionEncoding
from transformer_embedding import TransformerEmbedding
from torch.nn.functional import gelu
class EncoderStack(nn.Module):
    """
        position embedding - raw
        input embedding - how
        encoder - principle
        """
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_len: int):
        """
        Embedding: 1. num_embeddings
                   2. embedding_dim -> 300?
                   4. padding_idx
                   5. device
        """
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

    def forward(self, x):
        x = self.embedding(x) # visualize embedding
        output = self.encoder(x)
        return output