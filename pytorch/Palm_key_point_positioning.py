# -*- coding: utf-8 -*-

import sys

sys.path.append("/home/chenhaoran/MTCNN_TRAIN")

import cv2
# from tools.detect import create_mtcnn_net, MtcnnDetector
# import tools.vision as vision
from Palm_loc_and_cls.train_net.models import MyNet, LossFn
import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision
from torchvision import transforms


class PalmTest:
    def __init__(self, image_dir='', o_net_path=None):
        self.image_dir = image_dir
        self.o_net_path = o_net_path
        self.use_cuda = False
        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.57,), (0.18,)),

        ])
        self.landmarks = self.load_model()
        pass

    def parse_image_name(self, name):
        """
        解析图片的名称
        :param name:图片的名称
        :return:
        """
        name_name = name.split(';')[0]
        _ = name.split(';')[-1]
        name_format = _.split('-')[-1]
        landmark_list = _.split('-')[:-1]

        for idx, i in enumerate(landmark_list):
            landmark_list[idx] = (float(i.split(',')[0]), float(i.split(',')[1]))

        # 判断名称是否有问题
        if len(landmark_list) != 12:
            return False

        _landmark_list = []
        for idx, item in enumerate(landmark_list):
            _landmark_list.append(item[0])
            _landmark_list.append(item[1])
        landmark_list = _landmark_list

        return {'name': name_name,
                'landmark': landmark_list,
                'format': name_format}

    def load_model(self):
        """

        :return:
        """
        # TODO 1 加载模型
        use_cuda = self.use_cuda
        if self.o_net_path is not None:
            print('=======> loading')
            net = MyNet(use_cuda=False)
            net.load_state_dict(torch.load(self.o_net_path, map_location=torch.device('cpu')))
            if (use_cuda):
                net.to('cpu')
            net.eval()

        # TODO 2 准备好数据
        print(self.image_dir + '//crop_one.jpg')
        _img = Image.open(self.image_dir + '//crop_one.jpg')
        # parse_result = self.parse_image_name(item)
        # landmark = parse_result['landmark']
        # print('the gt landmark :', landmark)
        img = self.transforms(_img)
        # print(img.shape)
        img = img.unsqueeze(0)
        # print(img.shape)
        pred = net(img)
        # print(pred.shape)

        # landmark = np.array(landmark)
        # landmark = torch.from_numpy(landmark).float()
        # loss = LossFn().loss_landmark(pred, landmark / 192)
        pred = pred * 192
        # pred = pred.detach().numpy()

        # print('the pred landmark is :', pred)
        # print(loss)
        # print("=" * 20)
        # # print(pred.shape)
        # # print(landmark)
        #
        # self.visualize(_img, np.array(landmark))
        # self.visualize(_img, pred.detach().numpy())
        pred = pred.detach().numpy()[0]
        # for i in range(len(pred)):
        #     pred[i] = pred[i].item()
        # # print(pred)
        return pred

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
        # plt.figure('visualize')
        # plt.imshow(img)
        y = []
        x = []
        for i in range(12):
            x.append(int(landmark[2 * i]))
            y.append(int(landmark[2 * i + 1]))
        plt.plot(y, x, '*')
        # plt.show()


if __name__ == '__main__':
    test = PalmTest(o_net_path='./model_store/min_palm_model_epoch.pt')
