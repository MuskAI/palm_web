#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat May 26 13:57:59 2018

@author: wujiyang
"""

import sys
sys.path.append('./')
from models import MyNet
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class PalmTest:
    def __init__(self, o_net_path=None):
        self.image_dir = './align'
        self.o_net_path = o_net_path
        self.use_cuda = False
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.57,), (0.18,)),

        ])
        self.save_path = './pred_landmark'
        self.load_model()

        pass

    def parse_image_name(self, name):
        """
        解析图片的名称
        :param name:图片的名称
        :return:
        """
        name_name = name.split(';')[0]
        bbox_and_format = name.split(';')[-1]

        return {'name': name_name,
                'landmark_and_format': bbox_and_format}

    def load_model(self):
        """

        :return:
        """
        # TODO 1 加载模型
        use_cuda = self.use_cuda
        if self.o_net_path is not None:
            print('=======> loading')
            net = MyNet(use_cuda=False)
            net.load_state_dict(torch.load(self.o_net_path))
            if (use_cuda):
                net.to('cpu')
            net.eval()

        # TODO 2 准备好数据
        img_list = os.listdir(self.image_dir)
        for idx, item in enumerate(img_list):
            _img = Image.open(os.path.join(self.image_dir, item))
            parse_result = self.parse_image_name(item)
            landmark_and_format = parse_result['landmark_and_format']
            name = parse_result['name']
            img = self.transforms(_img)
            img = img.unsqueeze(0)

            pred = net(img)

            pred = pred * 192
            # pred = pred.detach().numpy()

            print('the pred landmark is :', pred)

            print("=" * 20)
            # # print(pred.shape)
            # # print(landmark)
            #
            try:
                self.save_pred(_img,name,landmark_and_format,pred.detach().numpy())
            # self.visualize(_img, np.array(landmark))
            # self.visualize(_img, pred.detach().numpy())
            # # print(pred)
            except:
                print('Error:',item)


    def visualize(self, img, landmark):
        """
        :param img: PIL Image 类型
        :param landmark: 一个list 其中每个点都是一个元组
        :return: 直接可视化结果
        """
        print('now we are in visualize')
        # print(landmark.shape)
        landmark = landmark.reshape(24)
        # print(landmark.shape)
        # print(landmark)
        img = np.array(img, dtype='uint8')
        plt.figure('visualize')
        plt.imshow(img)
        y = []
        x = []
        for i in range(12):
            x.append(int(landmark[2 * i]))
            y.append(int(landmark[2 * i + 1]))
        plt.plot(y, x, '*')
        plt.show()
    def save_pred(self,img,name,landmark_and_format,pred_landmark):
        """
        用来保存图片的pred
        :return:
        """
        # TODO 1 首先保存到文件名后
        img.save(os.path.join(self.save_path,name+'%.2f'*12 % tuple(pred_landmark)+';'+landmark_and_format))
if __name__ == '__main__':
    test = PalmTest(o_net_path='./model_store/min_palm_model_epoch.pt')

