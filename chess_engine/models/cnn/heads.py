"""
CNN ARCHITECTURE - Policy and Value Heads

Policy Head: Takes CNN features (256, 8, 8) and outputs move probabilities (4672,).
Value Head: Takes CNN features (256, 8, 8) and outputs position evaluation [-1, 1].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyHead(nn.Module):
    """1×1 conv + flatten + FC outputting move logits (batch, num_actions)."""

    def __init__(self, input_channels: int = 256, num_actions: int = 4672):
        super().__init__()

        self.num_actions = num_actions

        self.conv = nn.Conv2d(input_channels, 2, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(2)
        self.fc = nn.Linear(2 * 8 * 8, num_actions)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = x.flatten(1)
        x = self.fc(x)
        return x


class ValueHead(nn.Module):
    """1×1 conv + two FC layers + tanh, outputting position value in [-1, 1]."""

    def __init__(self, input_channels: int = 256):
        super().__init__()

        self.conv = nn.Conv2d(input_channels, 1, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 1)

        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x, inplace=True)
        x = x.flatten(1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.tanh(x)
        return x
