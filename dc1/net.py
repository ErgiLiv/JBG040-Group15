import torch
import torch.nn as nn

class CBAMBlock(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMBlock, self).__init__()
        # Channel Attention Module
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1),
            nn.Sigmoid()
        )
        # Spatial Attention Module
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(sa)
        x = x * sa

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),  # Added CBAM after first conv
            nn.MaxPool2d(kernel_size=4),
            nn.Dropout(p=0.5),

            nn.Conv2d(64, 32, kernel_size=4, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAMBlock(32),  # CBAM after second conv
            nn.MaxPool2d(kernel_size=3),
            nn.Dropout(p=0.25),

            nn.Conv2d(32, 16, kernel_size=4, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            CBAMBlock(16),  # CBAM after third conv
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=0.125),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(144, 256),
            nn.ReLU(),
            nn.Linear(256, 1)  # Binary classification
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
