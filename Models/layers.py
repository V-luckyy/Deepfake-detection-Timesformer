# layers.py
import torch
import torch.nn as nn
import math


# layers.py

class SpatialTemporalAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_patches):
        super(SpatialTemporalAttention, self).__init__()
        self.spatial_attn = SelfAttention(d_model, num_heads)  # 空间注意力
        self.temporal_attn = SelfAttention(d_model, num_heads)  # 时间注意力
        self.num_patches = num_patches
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x shape: (batch_size, num_frames, embedded_HW, d_model)
        batch_size, num_frames, num_patches, d_model = x.shape

        # 1. 时间维度的自注意力
        x_time = x.permute(0, 2, 1, 3)
        x_time = x_time.contiguous().view(batch_size * num_patches, num_frames, d_model)
        x_time = self.temporal_attn(x_time).view(batch_size, num_patches, num_frames, d_model)
        x_time = x_time.permute(0, 2, 1, 3)

        # 残差连接 + LayerNorm
        x = self.layernorm1(x + x_time)

        # 2. 空间维度的自注意力
        # reshape to (batch_size * num_frames, num_patches, d_model) for spatial attention
        x_space = x.view(batch_size * num_frames, num_patches, d_model)
        x_space = self.spatial_attn(x_space).view(batch_size, num_frames, num_patches, d_model)  # reshape back

        # 残差连接 + LayerNorm
        return self.layernorm2(x + x_space)


class PositionalEncoding3D(nn.Module):
    """
    正弦和余弦编码的几大优势如下：
    提供平滑的、可微、不同的周期的函数，便于模型通过线性变换解码位置信息。
    支持绝对位置和相对位置的建模，尤其是对于时空维度的数据，能够更好地捕捉相邻patch或帧的关系。
    交替结构避免了奇偶维度之间的耦合，增强了位置信息的独立性。
    正弦和余弦的周期性特性，使位置编码对输入位置的相对变化更鲁棒。
    只计算一半维度（d_model / 2），然后交替填充，既节约了内存，又不损失表达能力。
    """
    def __init__(self, d_model, num_frames, embedded_hw, num_patches, max_len=5000):
        super(PositionalEncoding3D, self).__init__()
        self.temporal_pe = torch.zeros(num_frames, d_model)
        self.spatial_pe = torch.zeros(num_patches, d_model)

        # Temporal position encoding
        temporal_position = torch.arange(0, num_frames).unsqueeze(1).float()    # (num_frames, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model / 2)
        self.temporal_pe[:, 0::2] = torch.sin(temporal_position * div_term)     # 对偶数列进行正弦位置编码 (广播相乘)
        self.temporal_pe[:, 1::2] = torch.cos(temporal_position * div_term)     # 对奇数列进行余弦位置编码

        # Spatial position encoding
        spatial_position = torch.arange(0, num_patches).unsqueeze(1).float()
        self.spatial_pe[:, 0::2] = torch.sin(spatial_position * div_term)
        self.spatial_pe[:, 1::2] = torch.cos(spatial_position * div_term)

        self.temporal_pe = self.temporal_pe.unsqueeze(0).unsqueeze(2)
        # (num_frames, d_model) -> (1, num_frames, 1, d_model)
        self.spatial_pe = self.spatial_pe.unsqueeze(0).unsqueeze(1)
        # (num_patches, d_model) -> (1, 1, num_patches, d_model)
        self.spatial_pe = self.spatial_pe.repeat_interleave(embedded_hw // num_patches, dim=2)
        # (1, 1, num_patches, d_model) -> (1, 1, em_height*em_width, d_model)

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_patches, d_model)
        # Repeat spatial position encoding for each patch

        return x + self.temporal_pe.to(x.device) + self.spatial_pe.to(x.device)


class SelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(SelfAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        batch_size, seq_len, d_model = x.size()

        # Compute Q, K, V
        qkv = self.qkv(x)  # (batch_size, seq_len, 3 * d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # Each of (batch_size, num_heads, seq_len, head_dim)

        # Scaled dot-product attention
        scores = torch.einsum("bhqd, bhkd -> bhqk", q, k) / math.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.einsum("bhqk, bhvd -> bhqd", attn_weights, v)

        # Concatenate heads and pass through final linear layer
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        return self.fc_out(attn_output)
