'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1_laser = nn.Conv2d(75, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_laser = nn.BatchNorm2d(64)

        self.layer1_laser = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2_laser = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3_laser = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4_laser = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear_laser = nn.Linear(8192, 4096)

        self.conv1_s = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_s = nn.BatchNorm2d(64)

        self.layer1_s = copy.deepcopy(self.layer1_laser)
        self.layer2_s = copy.deepcopy(self.layer2_laser)
        self.layer3_s = copy.deepcopy(self.layer3_laser)
        self.layer4_s = copy.deepcopy(self.layer4_laser)
        self.linear_s = nn.Linear(8192, 4096)

        self.transformer_1 = nn.Transformer(nhead=1, d_model=2048)
        self.transformer_2 = nn.Transformer(nhead=1, d_model=512)

        self.linear = nn.Linear(4096, 2048, bias=True)

        # self.bn1 = nn.BatchNorm2d(64)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # step 1
        laser = x[:, :75, :, :]

        laser = self.conv1_laser(laser)
        laser = self.bn1_laser(laser)
        laser = F.relu(laser)

        laser = self.layer1_laser(laser)
        laser = self.layer2_laser(laser)
        laser = self.layer3_laser(laser)

        laser = laser.reshape(2, 256, -1)

        s = x[:, 75:, :, :]
        s = self.conv1_s(s)
        s = self.bn1_s(s)
        s = F.relu(s)

        s = self.layer1_s(s)
        s = self.layer2_s(s)
        s = self.layer3_s(s)
        s = s.reshape(2, 256, -1)

        # step 2
        feature = torch.cat([laser, s], dim=-1)
        feature = self.transformer_1(feature, feature)

        laser_feature = feature[:, :, :1024]
        s_feature = feature[:, :, 1024:]

        laser = laser + laser_feature
        s = s + s_feature

        laser = laser.reshape(2, 256, 32, 32)
        s = s.reshape(2, 256, 32, 32)

        # step 3
        laser = self.layer4_laser(laser)
        s = self.layer4_s(s)

        laser = laser.reshape(2, 512, -1)
        s = s.reshape(2, 512, -1)

        feature = torch.cat([laser, s], dim=-1)
        feature = self.transformer_2(feature, feature)

        laser_feature = feature[:, :, :256]
        s_feature = feature[:, :, 256:]

        laser = laser + laser_feature
        s = s + s_feature

        # print("laser shape: ", laser.shape)
        # print("rf shape: ", rf.shape)
        # exit()

        laser = F.avg_pool2d(laser, 4)
        s = F.avg_pool2d(s, 4)

        laser = laser.reshape(laser.shape[0], -1)
        s = s.reshape(s.shape[0], -1)

        #
        # print("laser shape: ", laser.shape)
        # print("rf shape: ", rf.shape)
        # exit()

        # step 4
        laser = self.linear_laser(laser)
        s = self.linear_s(s)

        out = laser + s
        out = self.linear(out)
        # print("out: ", out.shape)
        # exit()
        out = out.reshape(2, 128, 4, 4)

        return out

def MultimodalFusionTransformer_2():
    return ResNet(BasicBlock, [2, 2, 2, 2])

if __name__ == "__main__":
    device = 'cuda:0'
    net = MultimodalFusionTransformer_2().cuda()
    B = 2



    laser_images = torch.Tensor(B, 5, 5, 3, 128, 64).to(device)
    laser_images = laser_images.reshape(B, 5*5*3, 128, 64)

    rf_data = torch.Tensor(B, 4, 128, 128).to(device)
    sound_data = torch.Tensor(B, 64, 257, 869).to(device)

    features = (laser_images, rf_data, sound_data)


    output = net(features)
    print("output shape: ", output.shape)