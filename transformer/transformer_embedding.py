import torch.nn as nn
import torch
from position_encoding import PositionEncoding
class TransformerEmbedding(nn.Module):
    """
    :param vocab size
    :param d_model: 768, 512
    :param max_len: 10000
    :param dropout: 0.1, 0.5
    :param device: 'cpu', 'cuda'
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1,
                 device: str = 'cpu'):
        super().__init__() # ?
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = PositionEncoding(d_model=d_model,
                                                   max_seq_len=max_len)
        self.dropout = nn.Dropout(p=dropout)

    # dim embedding? dim pos_embedding?
    def forward(self, x: torch.FloatTensor):
        """
        :param x:
        :return: dropout (embedding + position embedding)
        """
        token_embedding = self.embedding(x)
        pos_embedding = self.position_embedding(x)
        return self.dropout(token_embedding + pos_embedding)

if __name__ == "__main__":
    a = TransformerEmbedding(500, 768, 100)
    print(a.embedding.state_dict())