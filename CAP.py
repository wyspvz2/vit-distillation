import torch
import torch.nn as nn
import torch.optim as optim
import timm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import LayerNorm
from mmcv.cnn.bricks.transformer import MultiheadAttention
import numpy as np
from torch.nn import MultiheadAttention
from torchvision import datasets
import os

class CrossAttentionProjection(nn.Module):
    def __init__(self, student_dims, teacher_dims, num_heads=8, query_len=197):
        super(CrossAttentionProjection, self).__init__()

        self.student_dims = student_dims
        self.teacher_dims = teacher_dims
        self.num_heads = num_heads
        self.query_len = query_len  # Number of patches in each image (sequence length)

        # Linearly project the student features from student_dims to teacher_dims
        self.projection = nn.Linear(student_dims, teacher_dims)

        # Learnable query for cross-attention, shape [1, query_len, teacher_dims]
        self.query = nn.Parameter(torch.randn(1, query_len, teacher_dims))  # Use teacher_dims

        # Cross-attention setup, using teacher_dims for the embedding dimension for the query
        self.attention = MultiheadAttention(embed_dim=teacher_dims, num_heads=num_heads, batch_first=True)

        # LayerNorm and FFN for feature refinement
        self.norm = LayerNorm(teacher_dims)  # Use teacher_dims for layer norm
        self.ffn = nn.Sequential(
            nn.Linear(teacher_dims, teacher_dims * 4),
            nn.ReLU(),
            nn.Linear(teacher_dims * 4, teacher_dims)
        )

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor):
        """
        Args:
            student_features (torch.Tensor): Shape [B, S, student_dims], student's feature map
            teacher_features (torch.Tensor): Shape [B, S, teacher_dims], teacher's feature map
        """
        B, S, D = student_features.shape

        # Project student features to teacher_dims
        student_features_projected = self.projection(student_features)  # [B, S, teacher_dims]

        # Generate the query and repeat it for batch size
        query = self.query.repeat(B, 1, 1)  # [B, query_len, teacher_dims]

        # Projected student features are now used as keys and values
        student_features_projected = student_features_projected.view(B, S, self.teacher_dims)  # [B, S, teacher_dims]

        # Cross-attention: use the projected student features as keys and values, and the learnable query as queries
        attn_output, _ = self.attention(query, student_features_projected, student_features_projected)

        # Ensure the attn_output has the correct shape for reshaping: [B, S, teacher_dims]
        attn_output = attn_output.contiguous()  # Make sure the output is contiguous
        attn_output = attn_output.view(B, S, self.teacher_dims)  # [B, S, teacher_dims]

        # Add residual connection and layer normalization
        output = self.norm(attn_output + teacher_features)

        # FFN layer for further refinement
        output = self.ffn(output)

        return output


# CAP Module to handle cross-attention projections
class CAPModule(nn.Module):
    def __init__(self, student_dims, teacher_dims, num_heads=8, query_len=197):
        super(CAPModule, self).__init__()
        self.cross_attention = CrossAttentionProjection(student_dims, teacher_dims, num_heads, query_len)

    def forward(self, student_features: torch.Tensor, teacher_features: torch.Tensor):
        """
        Args:
            student_features (torch.Tensor): The intermediate features from the student network (ViT-base)
            teacher_features (torch.Tensor): The intermediate features from the teacher network (ViT-large)
        """
        # Generate pseudo-teacher features using cross-attention
        pseudo_teacher_features = self.cross_attention(student_features, teacher_features)

        return pseudo_teacher_features
