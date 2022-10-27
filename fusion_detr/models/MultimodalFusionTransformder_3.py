import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.MultimodalFusionTransformer import MultimodalFusionTransformer
from models.MultimodalFusionTransformer_2 import MultimodalFusionTransformer_2

class MultimodalFusionTransformer_3(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion_transformer_1 = MultimodalFusionTransformer()
        self.fusion_transformer_2 = MultimodalFusionTransformer_2()
        self.linear_proj = nn.Linear(2048, 2048, bias=True)

    def forward(self, x):
        laser = x[:, :75, :, :]
        rf = x[:, 75:79, :, :]
        sound = x[:, 79:, :, :]

        laser_rf = torch.cat([laser, rf], dim=1)
        laser_sound = torch.cat([laser, sound], dim=1)

        laser_rf = self.fusion_transformer_1(laser_rf)
        laser_sound = self.fusion_transformer_2(laser_sound)

        out = laser_rf + laser_sound
        out = out.reshape(2, -1)
        out = self.linear_proj(out)
        out = out.reshape(2, 128, 4, 4)

        return out
