#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:29:56 2018

@author: wujiyang
"""

import sys
from tensorboardX import SummaryWriter

sys.path.append('../')
writer = SummaryWriter(
    '../runs/' + '0207_02-08_tensorboard')
sys.path.append("/home/chenhaoran/MTCNN_TRAIN")

import os
import argparse
import datetime
import torch
import config
from torch.utils.data import Dataset
from train_net.models import LossFn, MyNet
from PIL import Image
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import numpy as np

class LandMarkRotation:
    def __init__(self):
        self.img_dir = '/home/liu/chenhaoran/after_resize2'
        pass

    def check(self):
        img_list = os.listdir(self.img_dir)
        for idx, item in enumerate(img_list):
            img_ = Image.open(os.path.join(self.img_dir, item))
            angle = [90,180,270]
            angle = random.sample(angle,1)[0]
            random_num = np.random.randint(-5, 5,1)[0]
            angle += random_num

            img = img_.rotate(-angle)
            parse_result = self.parse_image_name(item)
            landmark = parse_result['landmark']
            print('landmark:',landmark)
            self.visualize(img_, landmark)
            landmark = self.landmark_clockwise(landmark, angle)
            print('landmark:', landmark)
            self.visualize(img, landmark)
    def using_when_training(self,name):
        img_ = Image.open(os.path.join('/home/liu/chenhaoran/after_resize2', name))
        angle = [90, 180, 270]
        angle = random.sample(angle, 1)[0]
        random_num = np.random.randint(-5, 5, 1)[0]
        angle += random_num

        img = img_.rotate(-angle)
        parse_result = self.parse_image_name(name)
        landmark = parse_result['landmark']
        landmark = self.landmark_clockwise(landmark, angle)
        return img, landmark
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

    def visualize(self, img, landmark):
        """
        :param img: PIL Image 类型
        :param landmark: 一个list 其中每个点都是一个元组
        :return: 直接可视化结果
        """
        print('now we are in visualize')
        # print(landmark.shape)

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
        print(x)
        print(y)
        print()
        plt.plot(y, x, '*')
        plt.show()

    def landmark_clockwise(self, landmark, angle):
        """
        x1=xcos(β)-ysin(β);
        y1=ycos(β)+xsin(β);
        :param landmark:
        :return:
        """

        _landmark = []
        center_point = [96, 96]
        for i in range(12):
            y = landmark[2 * i]
            x = landmark[2 * i + 1]

            x1 = (x - center_point[0]) * round(np.cos(np.deg2rad(angle)), 2) - \
                 (y-center_point[1]) * round(np.sin(np.deg2rad(angle)), 2) + center_point[0]
            y1 = (x - center_point[0]) * round(np.sin(np.deg2rad(angle)), 2) + \
                 (y-center_point[1]) * round(np.cos(np.deg2rad(angle)), 2) + center_point[1]
            _landmark.append(y1)
            _landmark.append(x1)

        return _landmark


class PalmDataset(Dataset):
    def __init__(self, data_path, mode='train', val_split=0):
        self.mode = mode
        assert self.mode in ['train', 'val', 'test'], \
            "mode should be 'train' or 'test', but got {}".format(self.mode)
        self.img_dir = data_path
        self.img_list = os.listdir(self.img_dir)
        # 划分训练集和验证集合
        if self.mode in ['train', 'val']:
            random.seed(321)
            data_len = len(self.img_list)
            val_set_size = int(data_len * val_split)
            train_set_size = data_len - val_set_size
            if self.mode == 'val':
                self.data_img = random.sample(self.img_list, val_set_size)

            elif self.mode == 'train':
                self.data_img = random.sample(self.img_list, train_set_size)

        elif self.mode == 'test':
            pass

        self.transforms = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.57,), (0.18,)),

        ])

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
            try:
                landmark_list[idx] = (float(i.split(',')[0]), float(i.split(',')[1]))
            except:
                print('the error is ',landmark_list)

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

    # 每次迭代时返回数据和对应的标签
    def __getitem__(self, idx):
        # img,landmark = using_when_training(self.data_img[idx])
        # img = self.transforms(img)
        for i in range(5):
            try:

                img = Image.open(os.path.join(self.img_dir,self.data_img[idx]))
                img = self.transforms(img)
                # label = np.array(self.data_label.iloc[idx,:],dtype = 'float32')/96
                # print(self.data_img[idx])
                landmark = self.parse_image_name(self.data_img[idx])
                landmark = np.array(landmark['landmark'], dtype='float32')/192
                return {'img': img,
                        'landmark': landmark}

            except:
                print('error retry!')
                error_dealing = [1, 2, 3, -1, -2, -3]
                random_num = random.sample(error_dealing, 1)[0]
                idx = idx + random_num
            # visualize(np.array(img[0]),landmark*192)

    # 返回整个数据集的总数
    def __len__(self):
        return len(self.data_img)
def parse_image_name(name):
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
def landmark_clockwise(landmark, angle):
    """
    x1=xcos(β)-ysin(β);
    y1=ycos(β)+xsin(β);
    :param landmark:
    :return:
    """

    _landmark = []
    center_point = [96, 96]
    for i in range(12):
        y = landmark[2 * i]
        x = landmark[2 * i + 1]

        x1 = (x - center_point[0]) * round(np.cos(np.deg2rad(angle)), 2) - \
             (y-center_point[1]) * round(np.sin(np.deg2rad(angle)), 2) + center_point[0]
        y1 = (x - center_point[0]) * round(np.sin(np.deg2rad(angle)), 2) + \
             (y-center_point[1]) * round(np.cos(np.deg2rad(angle)), 2) + center_point[1]
        _landmark.append(y1)
        _landmark.append(x1)

    return _landmark


def train_o_net(annotation_file, model_store_path, end_epoch=1000, frequent=200, base_lr=0.01, batch_size=256,
                use_cuda=True):
    # initialize the ONet ,loss function and set optimization for this network
    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    net = MyNet(is_train=True, use_cuda=use_cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if use_cuda:
        net.to(device)
    lossfn = LossFn()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 100, 120,150,200], gamma=0.1)
    train_dataset = PalmDataset(data_path='/home/liu/chenhaoran/argument_data')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    # train net
    net.train()
    i=0
    min_loss = 100000
    for cur_epoch in range(end_epoch):
        scheduler.step(cur_epoch)
        per_epoch_loss = 0
        times = 0
        for batch_idx, item in enumerate(trainDataLoader):
            times +=1
            im_tensor = item['img']
            gt_landmark = item['landmark']

            # _check_im = np.array(im_tensor[0][0],dtype='uint8')
            # _check_gt = np.array(gt_landmark[0])*192
            # _check_im = Image.fromarray(_check_im)
            # visualize(_check_im,_check_gt)
            #
            # _check_im = np.array(im_tensor[1][0], dtype='uint8')
            # _check_gt = np.array(gt_landmark[1]) * 192
            # _check_im = Image.fromarray(_check_im)
            # visualize(_check_im, _check_gt)
            #
            # _check_im = np.array(im_tensor[2][0], dtype='uint8')
            # _check_gt = np.array(gt_landmark[2]) * 192
            # _check_im = Image.fromarray(_check_im)
            # visualize(_check_im, _check_gt)

            if use_cuda:
                # _check_im = np.array(im_tensor[0][0], dtype='uint8')
                im_tensor = im_tensor.cuda()
                gt_landmark = gt_landmark.cuda()

            pred = net(im_tensor)
            # if batch_idx == 1:
            #
            #     visualize(_check_im, pred[0].cpu().detach().numpy() * 192)
            landmark_loss = lossfn.landmark_loss(pred, gt_landmark)
            all_loss = landmark_loss
            per_epoch_loss = per_epoch_loss + float(all_loss.data.tolist())

            if batch_idx % frequent == 0:
                print(
                    "[%s, Epoch: %d, Step: %d] all_loss: %.6f landmark_loss: %.6f, lr: %.6f" %
                    (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), cur_epoch + 1, batch_idx,
                     all_loss.data.tolist(),
                     landmark_loss.data.tolist(), scheduler.get_lr()[0]))


            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()

        per_epoch_loss = per_epoch_loss/times
        # TODO: add validation set for trained model
        try:
            print(per_epoch_loss)
            writer.add_scalar('loss', per_epoch_loss, global_step=cur_epoch)
            if min_loss > float(per_epoch_loss):
                min_loss = float(per_epoch_loss)
                torch.save(net.state_dict(), os.path.join(model_store_path, "min_palm_model_epoch.pt"))
        except:
            pass
        if (cur_epoch + 1) % 10 == 0:
            torch.save(net.state_dict(), os.path.join(model_store_path, "!0208palm_model_epoch_%d_loss_%.6f.pt"
                                                      % (cur_epoch + 1, per_epoch_loss)))

    torch.save(net.state_dict(), os.path.join(model_store_path, '!0208palm_model_final.pt'))
def visualize(img, landmark):
    """
    :param img: PIL Image 类型
    :param landmark: 一个list 其中每个点都是一个元组
    :return: 直接可视化结果
    """
    print('now we are in visualize')
    # print(landmark.shape)

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
    print(x)
    print(y)
    print()
    plt.plot(y, x, '*')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description='Train ONet',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--anno_file', dest='annotation_file', help='training data annotation file',
                        default=os.path.join(config.ANNO_STORE_DIR, config.ONET_TRAIN_IMGLIST_FILENAME), type=str)
    parser.add_argument('--model_path', dest='model_store_path', help='training model store directory',
                        default=config.MODLE_STORE_DIR, type=str)
    parser.add_argument('--end_epoch', dest='end_epoch', help='end epoch of training',
                        default=1500, type=int)
    parser.add_argument('--frequent', dest='frequent', help='frequency of logging',
                        default=200, type=int)
    parser.add_argument('--base_lr', dest='base_lr', help='learning rate',
                        default=0.015, type=float)
    parser.add_argument('--batch_size', dest='batch_size', help='train batch size',
                        default=200, type=int)
    parser.add_argument('--gpu', dest='use_cuda', help='train with gpu',
                        default=config.USE_CUDA, type=bool)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    # print('train Onet argument:')
    # print(args)

    train_o_net(annotation_file=args.annotation_file, model_store_path=args.model_store_path,
                end_epoch=args.end_epoch, frequent=args.frequent, base_lr=args.base_lr, batch_size=args.batch_size,
                use_cuda=args.use_cuda)
