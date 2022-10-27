import torch
import torch.nn.functional as F
from torch import nn
from resnet import ResNet18, ResNet34
#
# class NlosDetectionModel(nn.Module):
#     def __init__(self):



if __name__ == "__main__":
    B = 2

    laser_images = torch.Tensor(B, 2, 5, 5, 128, 64, 3)
    rf_data = torch.Tensor(B, 128, 128, 4)
    sound_data = torch.Tensor(B, 257, 869, 64)

    # rgb_image = torch.Tensor(B, 720, 1280, 3)
    rgb_image = torch.Tensor(B, 3, 720, 1280)
    depth_image = torch.Tensor(B, 720, 1280)
    detection_gt = torch.Tensor(B, 2, 6)


    print("features")
    print("laser_image shape: ", laser_images.shape)
    print("rf_data shape: ", rf_data.shape)
    print("sound data shape: ", sound_data.shape)
    print("\n")

    print("targets")
    print("rgb_image shape: ", rgb_image.shape)
    print("depth_image shape: ", depth_image.shape)
    print("detection_gt shape: ", detection_gt.shape)

    image = torch.Tensor(B, 3, 256, 256)

    model = ResNet34()

    result = model(rgb_image)
    print("result shape: ", result.shape)
