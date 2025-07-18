import torch
import torch.nn as nn
import torch.nn.functional as F
from sim_matrix import similarity_fn_1, similarity_fn_2
class BiDirectionalAlignLayer(nn.Module):
    def __init__(self, dim, similarity_fn_1, similarity_fn_2, memory_tokens=197, num_heads=8):
        super().__init__()
        self.dim = dim
        self.similarity_fn_1 = similarity_fn_1
        self.similarity_fn_2 = similarity_fn_2
        self.memory_tokens = memory_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.opt_key = nn.Linear(dim, dim)
        self.opt_value = nn.Linear(dim, dim)
        self.sar_key = nn.Linear(dim, dim)
        self.sar_value = nn.Linear(dim, dim)

        self.memory_key = nn.Parameter(torch.randn(1, memory_tokens, num_heads, self.head_dim))
        self.memory_value = nn.Parameter(torch.randn(1, memory_tokens, num_heads, self.head_dim))

        self.query_proj = nn.Linear(dim, dim)

        self.memory_mapper = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, opt_feat, sar_feat, use_teacher=True):
        B, T, D = sar_feat.shape  # [B, 196, 768]

        # 学生特征映射key和value
        s_key = self.sar_key(sar_feat)      # [B, T, D]
        s_value = self.sar_value(sar_feat)  # [B, T, D]

        # 记忆key和value扩展
        mem_key = self.memory_key.expand(B, -1, -1, -1)      # [B, M, H, d]
        mem_value = self.memory_value.expand(B, -1, -1, -1)

        # 维度调整方便attention计算
        k = mem_key.permute(0, 2, 1, 3)  # [B, H, M, d]
        v = mem_value.permute(0, 2, 1, 3)  # [B, H, M, d]

        # 投影学生特征为query，分头处理
        query = self.query_proj(sar_feat).view(B, T, self.num_heads, self.head_dim)  # [B, T, H, d]
        query = query.permute(0, 2, 1, 3)  # [B, H, T, d]

        # 计算注意力分数
        attn_scores = torch.einsum("bhtd,bhmd->bhtm", query, k) / (self.head_dim ** 0.5)  # [B, H, T, M]
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 用注意力权重加权memory value得到每个token的记忆输出
        attn_output = torch.einsum("bhtm,bhmd->bhtd", attn_weights, v)  # [B, H, T, d]
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(B, T, D)  # [B, T, D]

        # 通过非线性映射生成伪教师特征
        pseudo_teacher_feat = self.memory_mapper(attn_output)  # [B, T, D]

        memory_align_loss = None
  
        if use_teacher:
            if opt_feat is None:
                raise ValueError("opt_feat must be provided when use_teacher=True")
            opt_feat = opt_feat.detach()  # 应该先 detach 原始教师特征，避免任何梯度传回去
            t_key_gt = self.opt_key(opt_feat)
            t_value_gt = self.opt_value(opt_feat)
            with torch.no_grad():
                opt_feat = opt_feat.detach()  # 防止它误被训练
            memory_align_loss = F.mse_loss(pseudo_teacher_feat, opt_feat)
            t_key_used = t_key_gt
            t_value_used = t_value_gt
        else:
            t_key_used = self.opt_key(pseudo_teacher_feat)
            t_value_used = self.opt_value(pseudo_teacher_feat)
        # SAR ← Teacher
        sim_matrix_1 = self.similarity_fn_1(s_key, t_key_used)
        sar_feat_updated = sar_feat + torch.bmm(sim_matrix_1, t_value_used)

        # Teacher ← SAR
        sim_matrix_2 = self.similarity_fn_2(t_key_used, s_key)
        opt_feat_fake = torch.bmm(sim_matrix_2, s_value)
        if opt_feat is not None:
            opt_feat_fake = opt_feat + opt_feat_fake
        else:
            opt_feat_fake = pseudo_teacher_feat + opt_feat_fake
    
        return opt_feat_fake, sar_feat_updated, memory_align_loss




class IterativeBiDirectionalAlign(nn.Module):
    def __init__(self, dim, similarity_fn_1, similarity_fn_2, num_iters=3, memory_tokens=197, num_heads=8):
        super().__init__()
        self.num_iters = num_iters
        self.layers = nn.ModuleList([
            BiDirectionalAlignLayer(dim, similarity_fn_1, similarity_fn_2, memory_tokens, num_heads)
            for _ in range(num_iters)
        ])

    def forward(self, opt_feat, sar_feat, use_teacher=True):
        opt_history = [opt_feat]
        sar_history = [sar_feat]

        total_memory_loss = 0.0

        for i in range(self.num_iters):
            opt_feat, sar_feat, memory_loss = self.layers[i](opt_feat, sar_feat, use_teacher=use_teacher)
            opt_history.append(opt_feat)
            sar_history.append(sar_feat)
            if memory_loss is not None:
                total_memory_loss += memory_loss

        if not use_teacher:
            total_memory_loss = None

        return opt_feat, sar_feat, opt_history, sar_history, total_memory_loss
