from typing import Dict

import torch
import torch.nn as nn


class TransformerLayer(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        ffn_dim_multiplier: int = 4,
        num_heads: int = 8,
        dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.1,
        activation="gelu",
    ) -> None:
        super().__init__()
        assert (
            feature_dim % num_heads == 0
        ), "Feature dimension must be dividable by number of heads."
        ffn_dim = ffn_dim_multiplier * feature_dim
        self.ffn1 = nn.Linear(feature_dim, ffn_dim)
        self.activation = getattr(nn.functional, activation)
        self.ffn2 = nn.Linear(ffn_dim, feature_dim)

        # Set batch_first=True for standard (batch, seq, feature) input format
        self.mha = nn.MultiheadAttention(
            feature_dim, num_heads, attention_dropout_rate, batch_first=True
        )

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.layernorm1 = nn.LayerNorm(feature_dim)
        self.layernorm2 = nn.LayerNorm(feature_dim)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # Post-LayerNorm Transformer Block
        # 1. Multi-head self-attention
        attn_output, attn_matrix = self.mha(
            query=x, key=x, value=x, attn_mask=attn_mask
        )
        attn_output = self.dropout1(attn_output)

        # 2. Residual connection + LayerNorm
        output_ln_1 = self.layernorm1(x + attn_output)

        # 3. Feed-forward network
        ffn_output = self.ffn1(output_ln_1)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn2(ffn_output)
        ffn_output = self.dropout2(ffn_output)

        # 4. Residual connection + LayerNorm
        output_ln_2 = self.layernorm2(ffn_output + output_ln_1)
        return {"output": output_ln_2, "attn_matrix": attn_matrix}
