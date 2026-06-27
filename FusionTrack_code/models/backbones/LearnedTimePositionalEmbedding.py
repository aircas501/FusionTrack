
import torch
import math
from torch import nn

from utils.nested_tensor import NestedTensor


class LearnedTimePositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added (batch_size, seq_len, d_model)
        """
        positions = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1))
        position_embeddings = self.embedding(positions)
        return x + position_embeddings


def build(config: dict):
    return LearnedTimePositionalEmbedding(d_model=config["REID_POS_HIDDEN_DIM"], max_len=config["REID_QUERY_LENGTH"])
