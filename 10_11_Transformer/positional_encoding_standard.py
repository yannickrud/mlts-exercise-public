'''
    File name: positional_encoding_standard.py
    Author: Richard Dirauf
    Python Version: 3.10
'''

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional Encoding Layer.
    Based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    Explaination: https://medium.com/@a.arun283/a-deeper-look-into-the-positional-encoding-method-in-transformer-architectures-7e98f32a925f
    """

    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        """Initialize Positional Encoding Layer.

        Args:
            d_model (int): Hidden dimension of the model
            dropout (float, optional): Optional dropout of layer. Defaults to 0.0.
            max_len (int, optional): Max. length of the PE. Defaults to 5000.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Positional Encoding Layer.

        Args:
            x (torch.Tensor): Input (seq_len, batch_size, embedding_dim)

        Returns:
            torch.Tensor: Output (seq_len, batch_size, embedding_dim)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
