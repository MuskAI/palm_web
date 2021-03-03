"""
@author HAORAN
time: 2021/2/19
更复杂的网络结构

"""
import torch
import torchvision
import torch.nn as nn
import numpy as np
from torchsummary import summary
import torch.nn.functional as F


class MyConv(nn.Module):
    def __init__(self, channels=3, kernel='filter1'):
        super(MyConv, self).__init__()
        self.channels = channels

        filter1 = [[-1, -1, -1],
                   [-1, 9, -1],
                   [-1, -1, -1]]

        filter2 = [[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]]
        filter3 = [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]

        filter4 = [[-1, -1, 0],
                   [-1, 0, 1],
                   [0, 1, 1]]

        filter5 = [[0, 1, 1],
                   [-1, 0, 1],
                   [-1, -1, 0]]
        if kernel == 'filter1':
            kernel = filter1
        elif kernel == 'filter2':
            kernel = filter2
        elif kernel == 'filter3':
            kernel = filter3
        elif kernel == 'filter4':
            kernel = filter4
        elif kernel == 'filter5':
            kernel = filter5
        else:
            print('kernel error')
            exit(0)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, self.channels, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


class PalmNet(nn.Module):
    def __init__(self, input_shape=(512, 512)):
        super(PalmNet, self).__init__()
        self.numberClass = 19
        self.enhancement = MyConv(kernel='filter1')
        self.prewitt1= MyConv(kernel='filter2')
        self.prewitt2 = MyConv(kernel='filter3')
        self.prewitt3 = MyConv(kernel='filter4')
        self.prewitt4 = MyConv(kernel='filter5')

        self.input_rgb = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.input_rgb_enhancement = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.prewitt_4cat = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.intput_to_densenet = nn.Sequential(

            nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        # self.densenet121 = torchvision.models.densenet121()
        self.densenet201 = torchvision.models.densenet201()
        self.outLayer1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5))
        self.outLayer2 = torch.nn.Linear(512, self.numberClass)


        # TODO 人工设定的卷积核

    def forward(self, rgb):
        rgb_enhancement = self.enhancement(rgb)
        prewitt_1 = self.prewitt1(rgb)
        prewitt_2 = self.prewitt2(rgb)
        prewitt_3 = self.prewitt3(rgb)
        prewitt_4 = self.prewitt4(rgb)

        prewitt = torch.cat([prewitt_1,prewitt_2,prewitt_3,prewitt_4],1)
        x_rgb = self.input_rgb(rgb)
        x_enhancement = self.input_rgb_enhancement(rgb_enhancement)
        x_prewitt_4cat = self.prewitt_4cat(prewitt)
        x = torch.cat([x_rgb, x_enhancement,x_prewitt_4cat,rgb], 1)


        # TODO 1 准备输入densenet
        x = self.intput_to_densenet(x)
        x = self.densenet201(x)
        x = self.outLayer1(x)
        x = self.outLayer2(x)
        return x


if __name__ == '__main__':
    model = PalmNet().cuda()
    summary(model, input_size=[[3, 512, 512]])
