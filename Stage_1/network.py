import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch.nn as nn
import torchvision.models as backbone_
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VGG_Network(nn.Module):
    def __init__(self, hp):
        super(VGG_Network, self).__init__()
        backbone = backbone_.vgg16(pretrained=True)  # vgg16, vgg19_bn

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        # if hp.pool_method is not None:
        #     self.pool_method = eval('nn.' + str(hp.pool_method) + '(1)')
        #     # AdaptiveMaxPool2d, AdaptiveAvgPool2d, AvgPool2d
        # else:
        #     self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default

        backbone.classifier._modules["6"] = nn.Linear(4096, 250)
        self.classifier = backbone.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Resnet_Network(nn.Module):
    def __init__(self, hp):
        super(Resnet_Network, self).__init__()
        backbone = backbone_.resnet18(pretrained=True)  # resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ["avgpool", "fc"]:
                self.features.add_module(name, module)

        self.pool_method = nn.AdaptiveMaxPool2d(1)  # as default

        if hp.dataset_name == "TUBerlin":
            num_class = 250
        else:
            num_class = 64
        num_class = 64
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, bb_box=None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# class ConvBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, userelu=True):
#         super(ConvBlock, self).__init__()
#         self.layers = nn.Sequential()
#         self.layers.add_module(
#             "Conv",
#             nn.Conv2d(
#                 in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
#             ),
#         )
#         self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))
#
#         if userelu:
#             self.layers.add_module("ReLU", nn.ReLU(inplace=True))
#
#         self.layers.add_module(
#             "MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
#         )
#
#     def forward(self, x):
#         out = self.layers(x)
#
#         return out
#
#
# class Resnet_Network(nn.Module):
#     def __init__(self, hp, opt=None):
#         super(Resnet_Network, self).__init__()
#
#         if opt is None:
#             opt = {
#                 "userelu": False,
#                 "in_planes": 3,
#                 "out_planes": [64, 64, 128, 128],
#                 "num_stages": 4,
#             }
#
#         self.in_planes = opt["in_planes"]
#         self.out_planes = opt["out_planes"]
#         self.num_stages = opt["num_stages"]
#         if type(self.out_planes) == int:
#             self.out_planes = [self.out_planes for i in range(self.num_stages)]
#         assert type(self.out_planes) == list and len(self.out_planes) == self.num_stages
#
#         num_planes = [
#             self.in_planes,
#         ] + self.out_planes
#         userelu = opt["userelu"] if ("userelu" in opt) else True
#
#         conv_blocks = []
#         for i in range(self.num_stages):
#             if i == (self.num_stages - 1):
#                 conv_blocks.append(
#                     ConvBlock(num_planes[i], num_planes[i + 1], userelu=userelu)
#                 )
#             else:
#                 conv_blocks.append(ConvBlock(num_planes[i], num_planes[i + 1]))
#         self.conv_blocks = nn.Sequential(*conv_blocks)
#
#         if hp.dataset_name == "TUBerlin":
#             num_class = 250
#         else:
#             num_class = 125
#         self.classifier = nn.Linear(32768, num_class)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2.0 / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#
#     def forward(self, x):
#         out = self.conv_blocks(x)
#         # print(out.shape)
#         x = torch.flatten(out, 1)
#         # print(x.shape)
#         x = self.classifier(x)
#         return x
#
