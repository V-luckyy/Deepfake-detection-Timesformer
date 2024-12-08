# model.py
import torch
import torch.nn as nn
from .layers import PositionalEncoding3D, SpatialTemporalAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, num_patches, dim_feedforward=2048, dropout=0.4):
        super(TransformerEncoderLayer, self).__init__()
        self.spatial_temporal_attn = SpatialTemporalAttention(d_model, num_heads, num_patches)
        self.layernorm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # Spatial-temporal attention block
        # src shape: (batch_size, seq_len, d_model)
        attn_output = self.spatial_temporal_attn(src)
        src = src + self.dropout(attn_output)
        src = self.layernorm1(src)

        # Feed-forward block
        ffn_output = self.ffn(src)
        src = src + self.dropout(ffn_output)
        return src


class VideoTransformer(nn.Module):
    def __init__(self, num_frames=16, num_patches=64, frame_size=128, d_model=512, num_heads=8, num_layers=2,
                 dim_feedforward=2048,num_classes=2):
        super(VideoTransformer, self).__init__()
        self.patch_embedding = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=d_model, kernel_size=(1, 5, 5), stride=(1, 3, 3)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3, 3), stride=(1, 2, 2)),
            nn.GELU(),
            # nn.Conv3d(in_channels=d_model, out_channels=d_model, kernel_size=(1, 3, 3), stride=(1, 2, 2),
            #           padding=(0, 1, 1))
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
        self.embedded_hw = ((((((frame_size-5)//3+1)//2)-3)//2+1)//2)**2
        self.positional_encoding = PositionalEncoding3D(d_model, num_frames, self.embedded_hw, num_patches)

        # Stack multiple Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, num_patches, dim_feedforward)
            for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        # Shape: (batch_size, channels, num_frames, height, width)
        x = self.patch_embedding(x)
        # Shape: (batch_size, d_model, num_frames, em_height, em_width)
        x = x.flatten(3).permute(0, 2, 3, 1)
        # Shape: (batch_size, num_frames, em_height*em_width, d_model)

        b, t, n, d = x.shape    # n: 64 d:512   b:32    t:16
        # 分别为 batch_size, num_frames, em_height*em_width, d_model
        x = self.positional_encoding(x)
        # Shape: (batch_size, num_frames, em_height*em_width, d_model)

        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Global average pooling and classification
        x = x.mean(dim=1)   # 时间维度池化，形状: (batch_size, num_patches, d_model)
        x = x.mean(dim=1)   # 空间维度池化，形状: (batch_size, d_model)
        return self.classifier(x)
