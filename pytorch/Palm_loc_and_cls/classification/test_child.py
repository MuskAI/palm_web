"""
@author haoran
time : 2/23
用于部署的小类测试程序

"""

import torch
import numpy as np
import torchvision.transforms as T
import sys

sys.path.append('../../../')
from PIL import Image
import cv2 as cv
import os
from .model_child_cls_plus import PalmNet
import traceback
import matplotlib.pylab as plt
import matplotlib

# matplotlib.use('TkAgg')
device = 'cpu'
def description(pred):
    # 便秘，胃十二指肠炎，胃神经功能症，附件炎，卵巢囊肿，乳腺增生，子宫肌瘤，胆结石，胆囊炎，慢性肝炎，脂肪肝，男性功能性障碍，前列腺炎，头晕，头痛，失眠，近视，咽炎，中耳炎耳鸣
    title = ['便秘', '胃十二指肠炎', '胃神经功能症', '附件炎', '卵巢囊肿', '乳腺增生', '子宫肌瘤', '胆结石', '胆囊炎', '慢性肝炎', '脂肪肝', '男性功能性障碍', '前列腺炎',
             '头晕', '头痛', '失眠', '近视', '咽炎', '中耳炎耳鸣']
    des_title = ['【分析结果】', '您有极大可能患有', '除此之外，掌纹信息中还反映出您有', '可能患有',
                 '的趋势']  # 有极大可能患有XXX，除此之外，掌纹信息中还反映出您有可能患有XXX和XXX的趋势
    des_big = []
    des_medium = []
    des_small = []
    for i in range(len(pred)):
        if pred[i] > 0.6:
            des_big.append(title[i])
        elif pred[i] > 0.3 and pred[i] <= 0.6:
            des_medium.append(title[i])
        elif pred[i] > 0.1 and pred[i] <= 0.3:
            des_small.append(title[i])
    des = '【分析结果】'
    if len(des_big) == 0 and len(des_medium) == 0 and len(des_small) == 0:
        des += '您的预检结果体现良好，但健康不容忽视，还是建议您定期去医院体检，将一切趋势扼杀在摇篮里。祝您身体健康。\n'
    else:
        if len(des_big) != 0:
            des += '您有极大可能患有'
            for i in range(len(des_big)):
                des += des_big[i]
                if i < len(des_big) - 2:
                    des += '、'
                elif i == len(des_big) - 2:
                    des += '和'
                else:
                    None
            des += '。\n'
        if len(des_medium) != 0 or len(des_small) != 0:
            if len(des_big) != 0:
                des += '除此之外，掌纹信息中还反映出您'
            else:
                des += '您在我们预测的疾病中，并未存在极大概率患有的疾病。\n但健康不容忽视，掌纹中还反映出一些其他信息，您'
            if len(des_medium) != 0:
                des += '有一定可能患有'
                for i in range(len(des_medium)):
                    des += des_medium[i]
                    if i < len(des_medium) - 2:
                        des += '、'
                    elif i == len(des_medium) - 2:
                        des += '和'
                    else:
                        None
                if len(des_small) != 0:
                    des += ',并存在极小可能患有'
                    for i in range(len(des_small)):
                        des += des_small[i]
                        if i < len(des_small) - 2:
                            des += '、'
                        elif i == len(des_small) - 2:
                            des += '和'
                        else:
                            None
                des += '。\n'
            else:
                if len(des_small) != 0:
                    des += '存在极小可能患有'
                    for i in range(len(des_small)):
                        des += des_small[i]
                        if i < len(des_small) - 2:
                            des += '、'
                        elif i == len(des_small) - 2:
                            des += '和'
                        else:
                            None
                    des += '。\n'
    des += '该结果由我们设计的AI算法，通过超过3万掌纹数据自动学习得出。以上预检结果并非绝对，但仍存在较高可信度。'
    if len(des_big) != 0 or len(des_medium) != 0 or len(des_small) != 0:
        des += '健康为先，您可以到权威医院进一步进行检查。'
    return des


class Diagnose:
    def __init__(self, img, model_path, output_path,img_name):
        """
        用于部署时候的小类疾病诊断类
        :param img: 输入是cv MAT
        :param model_path: 模型参数所在路径
        :param output_path: 诊断结果保存路径

        """

        self.model_path = model_path
        self.output_path = output_path
        self.img_name = img_name

        # TODO let img be the (1,3,512,512) tensor
        if True:
            img = np.array(img, dtype='uint8')
            r = Image.fromarray(img[:, :, 2]).convert('L')
            g = Image.fromarray(img[:, :, 1]).convert('L')
            b = Image.fromarray(img[:, :, 0]).convert('L')
            img = Image.merge("RGB", (r, g, b))

        img = T.Compose([
            T.ToTensor()
        ])(img)
        self.img = img.unsqueeze(0)


    def start(self):
        pred_sigmoid = self.test(self.img)
        self.deal_with_output(pred_sigmoid)
        des=''
        try:
            print('The input to description is :',list(pred_sigmoid[0]))
            des = description(pred=list(pred_sigmoid[0]))
            print('The output of the description',des)
        except Exception as e:
            print(e)
        return des

    def test(self, img):

        model_name = self.model_path
        model = PalmNet().cpu()
        model.eval()
        data = img
        # TODO 1 构建模型 数据加载 损失函数
        if not os.path.exists(model_name):
            traceback.print_exc('Please choose a right model path!!')
        else:
            checkpoint = torch.load(model_name, map_location='cpu')
            model.load_state_dict(checkpoint['state_dict'])
            print('==>loaded model:', model_name)
            image = data
            image = image.to(device)
            pred = model(image)
            pred_sigmoid = torch.nn.Sigmoid()(pred)
            print('The pred: ', np.where((pred.detach().numpy()) > 0.5, 1, 0))

            print('The pred_sigmoid: ', pred_sigmoid.detach().numpy())
            print('The pred_sigmoid: ', np.where((pred_sigmoid.detach().numpy()) > 0.5, 1, 0))
            print('=================================\n')

        return np.where(pred_sigmoid.detach().numpy()<0.05, 0, pred_sigmoid.detach().numpy())

    def deal_with_output(self, pred):
        child_cls_label = [1, 2, 3, 4, 5
            , 6, 7, 8, 9, 10
            , 11, 12, 13, 14, 15
            , 16, 17, 18, 19]
        child_cls_name = ['便秘', '胃十二指肠炎', '胃神经功能症', '附件炎',
                          '卵巢囊肿', '乳腺增生', '子宫肌瘤', '胆结石', '胆囊炎', '慢性肝炎',
                          '脂肪肝', '男性功能性障碍', '前列腺炎', '头晕', '头痛', '失眠',
                          '近视', '咽炎', '中耳炎耳鸣']
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.ylim(0,1)
        plt.bar(child_cls_label, list(pred[0]))
        plt.gcf().subplots_adjust(bottom=0.3)
        plt.legend()  # 显示图例，即label
        plt.xticks(ticks=child_cls_label, labels=child_cls_name, rotation=90)
        # plt.show()
        plt.savefig(os.path.join(self.output_path,self.img_name))
        plt.close()



if __name__ == '__main__':
    img = cv.imread('ML_1_1_1004;1,1-1,3-5,1-.jpg')

    Diagnose(img=img, model_path=r'D:\lab\fuse\0220model_epoch_144_0.148294.pt', output_path='/')
