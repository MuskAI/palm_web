#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:41:52 2018

@author: wujiyang
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchvision.models import resnet18

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


def compute_accuracy(prob_cls, gt_cls):
    '''return a tensor which contains predicted accuracy'''
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    # only positive and negative examples has the classification loss which labels 1 and 0
    mask = torch.ge(gt_cls, 0)
    valid_gt_cls = torch.masked_select(gt_cls,mask)
    valid_prob_cls = torch.masked_select(prob_cls,mask)
    # computer predicted accuracy
    size = min(valid_gt_cls.size()[0], valid_prob_cls.size()[0])
    prob_ones = torch.ge(valid_prob_cls,0.6).float()
    right_ones = torch.eq(prob_ones,valid_gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones),float(1.0)),float(size))


class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label)
        gt_label = torch.squeeze(gt_label)
        # only use negative samples and positive samples for classification which labels 0 and 1
        mask = torch.ge(gt_label, 0)
        valid_gt_label = torch.masked_select(gt_label, mask)
        valid_pred_label = torch.masked_select(pred_label, mask)
        return self.loss_cls(valid_pred_label, valid_gt_label) * self.cls_factor

    def box_loss(self, gt_label, gt_offset, pred_offset):
        pred_offset = torch.squeeze(pred_offset)
        gt_offset = torch.squeeze(gt_offset)
        gt_label = torch.squeeze(gt_label)
        # only use positive samples and partface samples for bounding box regression which labels 1 and -1
        unmask = torch.eq(gt_label,0)
        mask = torch.eq(unmask,0)
        #convert mask to dim index
        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)
        #only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, :]
        valid_pred_offset = pred_offset[chose_index, :]
        return self.loss_box(valid_pred_offset, valid_gt_offset) * self.box_factor
    #
    # def landmark_loss(self, gt_label, gt_landmark, pred_landmark):
    #     pred_landmark = torch.squeeze(pred_landmark)
    #     gt_landmark = torch.squeeze(gt_landmark)
    #     gt_label = torch.squeeze(gt_label)
    #     # only CelebA data been used in landmark regression
    #     mask = torch.eq(gt_label, -2)
    #
    #     chose_index = torch.nonzero(mask.data)
    #     chose_index = torch.squeeze(chose_index)
    #
    #     valid_gt_landmark = gt_landmark[chose_index, :]
    #     valid_pred_landmark = pred_landmark[chose_index, :]
    #     return self.loss_landmark(valid_pred_landmark,valid_gt_landmark) * self.land_factor
    #
    #

    def landmark_loss(self, pred_landmark, gt_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        # print()
        # print(pred_landmark[0])
        # print('-'*10)
        # print(gt_landmark[0])
        # print()
        return nn.MSELoss()(pred_landmark, gt_landmark)


class MyNet(nn.Module):
    ''' ONet '''

    def __init__(self, is_train=False, use_cuda=True):
        super(MyNet, self).__init__()
        num_keypoints = 12
        self.is_train = is_train
        self.use_cuda = use_cuda
        # backend
        self.in_process = torch.nn.Conv2d(in_channels=1, out_channels=3,kernel_size=3,padding=1)
        self.resnet18 = resnet18(pretrained=False)
        self.outLayer1 = torch.nn.Sequential(
            torch.nn.Linear(1000, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1))
        self.outLayer2 = torch.nn.Linear(512, num_keypoints*2)

    def forward(self, x):
        # backend
        x = self.in_process(x)
        x = self.resnet18(x)
        x = self.outLayer1(x)
        x = self.outLayer2(x)
        # x = self.pre_layer(x)
        # x = x.view(-1, 128 * 2 * 2 * 4 * 25)
        #
        # x = self.conv5(x)
        # x = self.prelu5(x)
        # # detection
        # # det = torch.sigmoid(self.conv6_1(x))
        # # box = self.conv6_2(x)
        # landmark = self.conv6_3(x)
            # return det, box, landmark

        return x

if __name__ == '__main__':
    # xx = torch.rand(2,3,320,320).cuda()
    model = MyNet().cuda()
    # writer.add_graph(model,xx)
    summary(model,(1,192,192))
    # print('ok')
    print('ok')