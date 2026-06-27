
import torch
import math
from torch import nn

from utils.nested_tensor import NestedTensor


class FixedTimePositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数索引使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数索引使用cos
        self.register_buffer('pe', pe)  # 注册为不参与学习的缓冲区

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            添加位置编码后的张量 (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:x.size(1)]  # 自动广播到batch维度
        return x


def build(config: dict):
    return FixedTimePositionalEncoding(d_model=config["REID_POS_HIDDEN_DIM"], max_len=config["REID_QUERY_LENGTH"])
