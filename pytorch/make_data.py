# import pandas as pd
import os
import shutil
import cv2
import numpy as np

train_save_path = ['D:\\pycharm\\palm area detection\\images\\train\\',
                   'D:\\pycharm\\palm area detection\\labels\\train\\',
                   'D:\\pycharm\\palm area detection\\gts\\train\\']
val_save_path = ['D:\\pycharm\\palm area detection\\images\\val\\',
                 'D:\\pycharm\\palm area detection\\labels\\val\\',
                 'D:\\pycharm\\palm area detection\\gts\\val\\']


# def read_csv(self, landmark_path):
#     """
#     读取CSV文件，转化为易于处理的字典形式:
#     {'img_name': '20001-女-27-左.jpg',
#     'landmark': [('2857', 1310.0), ('2204', 966.0), ('1776', 1160.0),
#     ('1330', 1335.0), ('1238', 1599.0), ('1208', 1664.0),
#      ('1239', 1903.0), ('1242', 1957.0), ('1388', 2193.0),
#      ('1441', 2243.0), ('1645', 2474.0), ('2910', 2221.0)]}
#
#     :return:
#     """
#     # read landmark file
#     df = pd.read_csv(landmark_path, encoding='gb2312', header=None)
#     print(df.shape[0])
#     # the image name index = 0 13 26;
#     name_index = [i * 13 for i in range(int(df.shape[0] / 13))]
#     image_and_landmark = []
#     for idx, item in enumerate(name_index):
#         x = list(df.loc[[item + 1, item + 2, item + 3, item + 4, item + 5, item + 6, item + 7, item + 8, item + 9,
#                          item + 10, item + 11, item + 12]][0])
#         y = list(df.loc[[item + 1, item + 2, item + 3, item + 4, item + 5, item + 6, item + 7, item + 8, item + 9,
#                          item + 10, item + 11, item + 12]][1])
#         loc = list(zip(x, y))
#
#         img_landmark_dict = {'img_name': str(df.loc[item][0]),
#                              'landmark': loc}
#         image_and_landmark.append(img_landmark_dict)
#
#     return image_and_landmark


def point_map(self, landmark, src_size, src_resize_size):
    """
    点映射，计算resize之后的坐标点
    :param landmark:
    :return:
    """
    # 缩小的倍数
    row_rate = src_size[0] / src_resize_size[0]
    col_rate = src_size[1] / src_resize_size[1]

    # 开始计算缩小后的坐标点的位置
    x, y = self.unzip(landmark=landmark)
    for i in range(12):
        x[i] = int(x[i] / col_rate)
        y[i] = int(y[i] / row_rate)

    return list(zip(x, y))


def path_mod(i, data_len):
    '''
    分配为训练集还是验证集，9:1
    :param i: 第i张图片
    :param data_len: 总长度
    :return: image保存路径，label保存路径，gt保存路径
    '''
    if i < data_len * 0.9:
        image_save_path = train_save_path[0]
        label_save_path = train_save_path[1]
        gt_save_path = train_save_path[2]
    else:
        image_save_path = val_save_path[0]
        label_save_path = val_save_path[1]
        gt_save_path = val_save_path[2]
    return image_save_path, label_save_path, gt_save_path


def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def my_cv_imwrite(filepath, img):
    # 使用imencode函数进行读取
    cv2.imencode('.jpg', img)[1].tofile(filepath)


def cal_xyxy(name):
    '''
    从命名中获取12个关键点
    :param name: 命名
    :return: 左上右下4个边界
    '''
    # image = my_cv_imread(read_name)
    name = name[:-5].split(';', 1)[1]
    name = name.split('-')
    name = [name[i].split(',') for i in range(12)]
    for i in range(len(name)):
        for j in range(len(name[i])):
            name[i][j] = int(name[i][j])
        # cv2.circle(image, (name[i][1], name[i][0]), 1, (0, 0, 255), 4)
    up = min(name, key=lambda x: x[0])[0]
    down = max(name, key=lambda x: x[0])[0]
    left = min(name, key=lambda x: x[1])[1]
    right = max(name, key=lambda x: x[1])[1]
    return left, up, right, down
    # cv2.rectangle(image, (left, up), (right, down), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    # cv2.imshow('a', image)
    # cv2.waitKey(0)


def xyxy2xywh(xyxy):
    '''
    :param xyxy: (x,y,x,y)左上角，右下角
    :return: (x,y,w,h)中心点，宽，长
    '''
    left, up, right, down = xyxy
    h = down - up
    w = right - left
    x = (left + right) / 2
    y = (up + down) / 2
    return x, y, w, h


def xywh2xyxy(xywh):
    '''
    :param xywh: (x,y,w,h)中心点，宽，长
    :return: (x,y,x,y)左上角，右下角
    '''
    x, y, w, h = xywh
    down = (2 * y + h) / 2
    up = (2 * y - h) / 2
    right = (2 * x + w) / 2
    left = (2 * x - w) / 2
    return left, up, right, down


def plot_area(read_name, write_name, xyxy, xywh):
    '''
    读取图片，画出检测区域并保存
    :param read_name: 需要读取图片的路径
    :param write_name: 需要保存图片的路径
    :param xyxy: 左上角和右下角
    :param xywh: 中心点和长宽
    :return:
    '''
    c1 = (xyxy[0], xyxy[1])
    c2 = (xyxy[2], xyxy[3])
    center = (xywh[0], xywh[1])
    image = my_cv_imread(read_name)
    cv2.circle(image, center, 1, (0, 0, 255), 4)
    cv2.rectangle(image, c1, c2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.imencode('.jpg', image)[1].tofile(write_name)


def normalization(size, data):
    '''
    对数据进行归一化
    :param size: 图片尺寸
    :param data: 需要归一化的数据
    :return: 归一化后的数据
    '''
    dw = 1 / size
    for i in range(len(data)):
        data[i] = data[i] * dw
    return data


def discover(size, data):
    '''
    反归一化后
    :param size: 图片尺寸
    :param data: 归一化后的数据
    :return:映射到图片的数据
    '''
    data = [int(data[i] * size) for i in range(len(data))]
    return data


def train_data_generation():
    '''
    主函数，用于生成训练数据（images和labels），将图片分为训练集，测试集
    :return:
    '''
    data_path = 'E:\\after_resize'
    data_list = os.listdir(data_path)
    data_list.sort(key=lambda x: int(x[:5]))
    data_len = len(data_list)

    # train:val=9:1
    for i in range(data_len):
        image_save_path, label_save_path, gt_save_path = path_mod(i, data_len)
        name = data_list[i]
        xyxy = cal_xyxy(name)
        xywh = xyxy2xywh(xyxy)
        x, y, w, h = normalization(192, xywh)
        shutil.copy(data_path + '\\' + name, image_save_path)
        with open(label_save_path + name[:-4] + '.txt', "w") as f:
            f.write('0 ' + format(x, '.6f') + ' ' + format(y, '.6f') + ' '
                    + format(w, '.6f') + ' ' + format(h, '.6f'))
        plot_area(image_save_path + name, gt_save_path + name, xyxy, xywh)


def cal_xywh(string):  # 左上点
    '''
    作crop使用，计算左上角和长宽
    :param string: label中的字符串
    :return: xywh为读取到的中心点和长宽，xyxy为xywh转换后的左上角和右下角，(x,y,w,h)为左上角和长宽
    '''
    xywh = read_labels(string)
    xyxy = xywh2xyxy(xywh)
    x = discover(192, xyxy)[0]
    y = discover(192, xyxy)[1]
    w = discover(192, xywh)[2]
    h = discover(192, xywh)[3]
    return xywh, xyxy, (x, y, w, h)


def read_labels(string):
    '''
    对预测的xywh进行读取，并转换为float
    :param string: 从label中读取的字符串
    :return: (x,y,w,h)
    '''
    xywh = string.split(' ')[1:]
    for i in range(len(xywh)):
        xywh[i] = float(xywh[i])
    return xywh


def save_crop_img(read_path, save_path, xywh_zs):
    '''
    对图片进行crop并保存
    :param read_path:需要读取图片的路径
    :param save_path:需要保存图片的路径
    :param xywh_zs:（x,y,w,h）其中x,y为左上角的坐标
    :return:空
    '''
    x, y, w, h = xywh_zs
    img = my_cv_imread(read_path)
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    # cv2.imshow('b', img)
    # cv2.waitKey(0)
    crop_img = img[y - 1:y + 1 + h, x - 1:x + 1 + w]
    # cv2.imshow('a', crop_img)
    # cv2.waitKey(0)
    cv2.imencode('.jpg', crop_img)[1].tofile(save_path)


def crop_data_generation():
    '''
    主函数，生成crop后图像
    :return:
    '''
    data_path = 'D:\\pycharm\\palm area detection\\images\\val\\'
    label_path = 'D:\\pycharm\\yolov5-master\\watch\\detect\\exp5\\labels\\'
    crop_path = 'D:\\pycharm\\yolov5-master\\watch\\detect\\exp5\\crop\\'
    data_list = os.listdir(data_path)
    data_list.sort(key=lambda x: int(x[:5]))
    data_len = len(data_list)
    for i in range(data_len):
        name = data_list[i]
        with open(label_path + name[:-4] + '.txt', 'r') as f:
            string = f.readlines()
        if len(string) == 1:
            xywh, xyxy, xywh_zs = cal_xywh(string[0])
            save_crop_img(data_path + name, crop_path + name[:-4] + '1.jpg', xywh_zs)
        else:
            for j in range(len(string)):
                xywh, xyxy, xywh_zs = cal_xywh((string[j]))
                save_crop_img(data_path + name, crop_path + name[:-4] + str(j) + '.jpg', xywh_zs)


def detect_data_resize():
    '''
    主函数，对图片进行resize
    :return:
    '''
    big_kinds = ['1肠胃', '2肺病', '3妇科', '4肝胆系统疾病', '5男科', '6内分泌疾病', '7神经系统', '8五官', '9心脑血管疾病', '10其他小类']
    data_path = 'E:\\Palam_Data_AfterClear\\'
    resize_data_path = 'D:\\pycharm\\Palam_Data_AfterClear_Afterresize\\'
    error_path = 'D:\\pycharm\\Palam_Data_AfterClear_Afterresize\\error.txt'
    for i in range(len(big_kinds)):
        big_disease_path = data_path + big_kinds[i] + '\\'
        resize_big_disease_path = resize_data_path + big_kinds[i] + '\\'
        if not os.path.exists(resize_big_disease_path):
            os.makedirs(resize_big_disease_path)
        small_kinds = os.listdir(big_disease_path)
        for i in range(len(small_kinds)):
            small_disease_path = big_disease_path + small_kinds[i] + '\\'
            resize_small_disease_path = resize_big_disease_path + small_kinds[i] + '\\'
            if not os.path.exists(resize_small_disease_path):
                os.makedirs(resize_small_disease_path)
            data_list = os.listdir(small_disease_path)
            for i in range(len(data_list)):
                name = data_list[i]
                try:
                    img = my_cv_imread(small_disease_path + name)
                    img = cv2.resize(img, (192, 192))
                    my_cv_imwrite(resize_small_disease_path + name, img)
                except Exception as e:
                    print(e)
                    print(small_disease_path + name)
                    with open(error_path, 'w') as f:
                        f.write(small_disease_path + name)


def detect_data_xywh2xyxy():
    '''
    预测到的为中心点和长宽，转换为左上和右下
    :return:
    '''
    big_kinds = ['1肠胃', '2肺病', '3妇科', '4肝胆系统疾病', '5男科', '6内分泌疾病', '7神经系统', '8五官', '9心脑血管疾病', '10其他小类']
    xywh_labels_path = 'D:\\pycharm\\yolov5-master\\runs\\detect\\Palam_Data_AfterClear_xywh\\'
    xyxy_labels_path = 'D:\\pycharm\\yolov5-master\\runs\\detect\\Palam_Data_AfterClear_xyxy\\'
    error_path = 'D:\\pycharm\\yolov5-master\\runs\\detect\\Palam_Data_AfterClear_xywh\\error.txt'
    for i in range(len(big_kinds)):
        xywh_big_disease_path = xywh_labels_path + big_kinds[i] + '\\'
        xyxy_big_disease_path = xyxy_labels_path + big_kinds[i] + '\\'
        if not os.path.exists(xyxy_big_disease_path):
            os.makedirs(xyxy_big_disease_path)
        small_kinds = os.listdir(xywh_big_disease_path)
        small_kinds.remove('zip')
        for i in range(len(small_kinds)):
            xywh_small_disease_path = xywh_big_disease_path + small_kinds[i] + '\\'
            xyxy_small_disease_path = xyxy_big_disease_path + small_kinds[i] + '\\'
            if not os.path.exists(xyxy_small_disease_path):
                os.makedirs(xyxy_small_disease_path)
            data_list = os.listdir(xywh_small_disease_path + 'labels\\')
            for i in range(len(data_list)):
                name = data_list[i]
                try:
                    with open(xywh_small_disease_path + 'labels\\' + name, 'r') as f:
                        string = f.readlines()
                    if len(string) == 1:
                        x, y, w, h = read_labels(string[0])
                        x1, y1, x2, y2 = xywh2xyxy((x, y, w, h))
                        print(format(x1, '.6f'))
                        with open(xyxy_small_disease_path + name, 'w') as h:
                            h.write(format(x1, '.6f') + ' ' + format(y1, '.6f') + ' '
                                    + format(x2, '.6f') + ' ' + format(y2, '.6f'))
                    else:
                        for j in range(len(string)):
                            x1, y1, x2, y2 = read_labels(string[j])
                            with open(xyxy_small_disease_path + name, 'a') as h:
                                h.write(format(x1, '.6f') + ' ' + format(y1, '.6f') + ' '
                                        + format(x2, '.6f') + ' ' + format(y2, '.6f'))
                except Exception as e:
                    print(e)
                    print(xywh_small_disease_path + name)
                    with open(error_path, 'a') as f:
                        f.write(xywh_small_disease_path + name)


def xywh2xyxy_(txt_path):
    '''
    预测到的为中心点和长宽，转换为左上和右下
    :return:
    '''
    xywh_labels_path = txt_path
    xyxy_labels_path = txt_path[:-4] + '_xyxy' + '.txt'
    with open(xywh_labels_path, 'r') as f:
        string = f.readlines()
    x, y, w, h = read_labels(string[0])
    x1, y1, x2, y2 = xywh2xyxy((x, y, w, h))
    with open(xyxy_labels_path, 'w') as h:
        h.write(format(x1, '.6f') + ' ' + format(y1, '.6f') + ' '
                + format(x2, '.6f') + ' ' + format(y2, '.6f'))
    # else:
    #     for j in range(len(string)):
    #         x1, y1, x2, y2 = read_labels(string[j])
    #         x1, y1, x2, y2 = xywh2xyxy((x, y, w, h))
    #         with open(xyxy_labels_path, 'a') as h:
    #             h.write(format(x1, '.6f') + ' ' + format(y1, '.6f') + ' '
    #                     + format(x2, '.6f') + ' ' + format(y2, '.6f'))
    return [x1, y1, x2, y2]


if __name__ == '__main__':
    detect_data_xywh2xyxy()
