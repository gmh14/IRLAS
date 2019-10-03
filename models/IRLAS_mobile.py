from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input

    def adjust_keep_prob(self, ratio):
        pass


class Block(nn.Module):
    def __init__(self, in_channel, channel):
        super(Block, self).__init__()

        self.op1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
        )
        if in_channel != channel:
            self.op2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channel, channel, 1),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 3, padding=1, groups=channel, bias=False),
                nn.Conv2d(channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
            )
            self.op3 = nn.Sequential(
                nn.Conv2d(channel, in_channel, kernel_size=(1, 1), stride=(1, 1), bias=False),
                nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True),
            )
            self.op4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channel, channel, 1),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(channel, channel, 5, padding=2, groups=channel, bias=False),
                nn.Conv2d(channel, channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
            )
        else:
            self.op2 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channel, in_channel, 3, padding=1, groups=in_channel, bias=False),
                nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True),
            )
            self.op3 = Identity()
            self.op4 = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_channel, in_channel, 5, padding=2, groups=in_channel, bias=False),
                nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(in_channel, eps=1e-05, momentum=0.1, affine=True),
            )

        self.op5 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, in_channel, 1, groups=in_channel, bias=False),
            nn.Conv2d(in_channel, channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channel, eps=1e-05, momentum=0.1, affine=True),
        )



    def forward(self, x):
        x0 = x
        x1 = self.op1(x0)
        x2 = self.op2(x0)
        x3 = x0 + self.op3(x2)
        x4 = self.op4(x0)
        x5 = self.op5(x3)
        x = torch.cat([x1, x4, x5], 1)
        return x


class IRLAS(nn.Module):
    def __init__(self):
        super(IRLAS, self).__init__()
        blocks_num = [1, 1, 4, 1]
        channels = [2, 8, 16, 32]
        Hin = 3
        self.num_classes = 1000

        self.conv1 = nn.Conv2d(Hin, 16 * channels[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(16 * channels[0], eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16 * channels[0], 16 * channels[0], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
                nn.BatchNorm2d(16 * channels[0], eps=1e-05, momentum=0.1, affine=True),
            ),
            Block(16 * channels[0], 16 * channels[0]),
        )
        self.layer2 = nn.Sequential(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16 * channels[0] * 3, 16 * channels[1], kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(16 * channels[1], eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(16 * channels[1], 16 * channels[1], kernel_size=(3, 3), stride=(2, 2),
                          padding=(1, 1)),
                nn.BatchNorm2d(16 * channels[1], eps=1e-05, momentum=0.1, affine=True,
                               ),
            ),
            Block(16 * channels[1], 16 * channels[1]),
        )
        self.layer3 = nn.Sequential(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16 * channels[1] * 3, 16 * channels[2], kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(16 * channels[2], eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(16 * channels[2], 16 * channels[2], kernel_size=(3, 3), stride=(2, 2),
                          padding=(1, 1)),
                nn.BatchNorm2d(16 * channels[2], eps=1e-05, momentum=0.1, affine=True),
            ),
            Block(16 * channels[2], 16 * channels[2]),
            Block(16 * channels[2] * 3, 16 * channels[2]),
            Block(16 * channels[2] * 3, 16 * channels[2]),
            Block(16 * channels[2] * 3, 16 * channels[2]),
        )
        self.layer4 = nn.Sequential(
            nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(16 * channels[2] * 3, 16 * channels[3], kernel_size=(1, 1), stride=(1, 1)),
                nn.BatchNorm2d(16 * channels[3], eps=1e-05, momentum=0.1, affine=True),
                nn.ReLU(),
                nn.Conv2d(16 * channels[3], 16 * channels[3], kernel_size=(3, 3), stride=(2, 2),
                          padding=(1, 1)),
                nn.BatchNorm2d(16 * channels[3], eps=1e-05, momentum=0.1, affine=True),
            ),
            Block(16 * channels[3], 16 * channels[3]),
        )
        self.avgpool = nn.AvgPool2d(kernel_size=[7, 7], stride=[7, 7], padding=0)

        self.fc = nn.Linear(16 * channels[3] * 3, self.num_classes)

        self.netpara = sum(p.numel() for p in self.parameters()) / 1e6

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
