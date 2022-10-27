# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from tokenize import group

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding
from models.MultimodalFusionTransformer import MultimodalFusionTransformer
from models.MultimodalFusionTransformer_2 import MultimodalFusionTransformer_2
from models.MultimodalFusionTransformder_3 import MultimodalFusionTransformer_3
from models.resnet_gn_4d import resnet18 as resnet18_4d
from models.resnet_gn import resnet18

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool, sensor: str):
        super().__init__()
        if sensor is not None:
            self.multimodal = True
        else:
            self.multimodal = False

        if self.multimodal:
            self.body = backbone
        else:
            for name, parameter in backbone.named_parameters():
                if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                    parameter.requires_grad_(False)
            if return_interm_layers:
                return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            else:
                return_layers = {'layer4': "0"}
            self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        self.sensor = sensor
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        if self.multimodal:
            xs = {"0": self.body(tensor_list.tensors)}
        else:
            xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out
    
class MyGroupNorm(nn.Module):
    def __init__(self, num_channels):
        super(MyGroupNorm, self).__init__()
        self.norm = nn.GroupNorm(num_groups=8, num_channels=num_channels,
                                 eps=1e-5, affine=True)
    
    def forward(self, x):
        x = self.norm(x)
        return x

class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 sensor: str):


        if sensor == "laser_4d":
            backbone = resnet18_4d(pretrained=False, group_norm=8, in_channel=6)
        elif sensor == "laser":
            backbone = resnet18(pretrained=False, group_norm=8, in_channel=150)
        elif sensor == "rf":
            backbone = resnet18(pretrained=False, group_norm=8, in_channel=4)
        elif sensor == "sound":
            backbone = resnet18(pretrained=False, group_norm=8, in_channel=64, is_sound = True)
        

        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        if sensor == "laser_rf" or sensor == "laser_sound" or sensor == "laser_rf_sound":
            num_channels = 128
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers, sensor)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        # print("Joiner: ", self)
        # print("tensor_list: ", tensor_list.tensors.shape)
        # print("self[0]: ", self[0])
        # exit()
        xs = self[0](tensor_list)
        # print("xs: ", xs['0'].tensors.shape)
        # exit()
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.sensor)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
