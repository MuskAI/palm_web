import cv2
import matplotlib.pyplot as plt
import os
import numpy as np


def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def my_cv_imwrite(filepath, img):
    # 使用imencode函数进行读取
    cv2.imencode('.jpg', img)[1].tofile(filepath)


def name_process(name):
    '''
    ML_1_1_1;172.98, 49.00, 112.31, 8.51, 67.28, 36.89, 19.22, 50.27, 10.90, 81.38, 11.60, 87.70, 10.94, 118.05, 12.84, 122.93, 22.64, 148.26, 33.55, 155.79, 55.53, 182.33, 181.54, 148.60,;
    503, 1711, 2502, 3683.jpg
    把name用；分割为三部分，第一部分为图片本身命名ML_1_1_1，第二部分为12个关键点，第三部分为手掌检测区域的左上和右下两点坐标
    再通过_,分别对三部分进行细分
    其中把第一部分和第三部分的数值全都转换为int，把第二部分的述职全都转换为float
    :param name:图片命名
    :return: {'1': ['ML_10_1_10', ['ML', 10, 1, 10]],
                '2': ['189.06,44.72,52.62,9.56,31.74,38.61,25.67,39.75,12.74,70.23,9.17,71.75,8.66,105.93,7.54,113.15,17.82,145.63,66.06,157.18,120.36,181.52,175.38,151.15,', [[189, 44], [52, 9], [31, 38], [25, 39], [12, 70], [9, 71], [8, 105], [7, 113], [17, 145], [66, 157], [120, 181], [175, 151]]],
                '3': ['614,1806,2330,3570.jpg', [614, 1806, 2330, 3570]]}
    '''
    name = name.split(';', 2)
    name_after = {'1': [name[0]], '2': [name[1]], '3': [name[2]]}
    name_part1 = name[0].split('_')
    name_part2 = name[1].split(',')
    name_part3 = name[2][:-4].split(',')
    for i in range(1, 4):
        name_part1[i] = int(name_part1[i])
    name_part2 = [[float(name_part2[2 * i]), float(name_part2[2 * i + 1])] for i in range(12)]
    name_part3 = [int(name_part3[i]) for i in range(4)]
    name_after['1'].append(name_part1)
    name_after['2'].append(name_part2)
    name_after['3'].append(name_part3)
    return name_after


def Left_or_Right(img, KeyPoints):
    '''
    检测手掌是左手还是右手，因为所有图像的关键点都是顺时针的方向，所以如果碰到右手还需要把关键点逆序
    还需要把右手进行对称变成左手，关键点也需要在x轴上进行变换
    :param img: 输入图片
    :param KeyPoints:12个关键点
    :return: 输出图片
    '''
    b1 = (KeyPoints[0][0] - KeyPoints[1][0]) / (KeyPoints[0][0] - KeyPoints[2][0])
    b2 = (KeyPoints[11][0] - KeyPoints[10][0]) / (KeyPoints[11][0] - KeyPoints[9][0])
    if (b1 < b2):
        return img, KeyPoints
    else:
        img = cv2.flip(img, 1, dst=None)
        for i in range(12):
            KeyPoints[i][1] = img.shape[1] - KeyPoints[i][1]
        KeyPoints.reverse()
        # my_cv_imwrite(save_path + 'turn_right.jpg', img)
        return img, KeyPoints


def line12_78(img, KeyPoints):
    '''
    计算12的中心点和78的中心点的连线
    :param img: 输入图片
    :param KeyPoints: 12个关键点
    :return: dx,dy和中心点
    '''
    c1 = (int((KeyPoints[0][1] + KeyPoints[11][1]) / 2), int((KeyPoints[0][0] + KeyPoints[11][0]) / 2))
    c2 = (int((KeyPoints[6][1] + KeyPoints[7][1]) / 2), int((KeyPoints[6][0] + KeyPoints[7][0]) / 2))
    # img = cv2.line(img, c1, c2, (0, 0, 255), 10)
    # plt.imshow(img)
    # plt.show()
    center = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    return dx, dy, center


def warp_affine(img, tmp):
    '''
    计算角度，用于旋转和仿射变换
    :param img: 输入图像
    :param tmp: (dx, dy, center)
    :return: 矫正后的图像
    '''
    dx, dy, center = tmp
    angle = cv2.fastAtan2(dy, dx)
    rot = cv2.getRotationMatrix2D(center, angle + 90, scale=1.0)
    rot_img = cv2.warpAffine(img, rot, dsize=(img.shape[1], img.shape[0]))
    # my_cv_imwrite(save_path + 'warp_affine.jpg', img)
    return rot_img


def draw_polygan(img, KeyPoints):
    '''
    :param img:crop后的图像
    :param KeyPoints: 12个关键点
    :return: 根据12个关键点对背景置黑后的图像
    '''
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    area = np.array(KeyPoints)[:, ::-1]
    # area = np.array(
    #     [KeyPoints[0][::-1], KeyPoints[1][::-1], KeyPoints[2][::-1], KeyPoints[3][::-1], KeyPoints[10][::-1],
    #      KeyPoints[11][::-1]])
    cv2.fillPoly(mask, [area], (255, 255, 255))
    # plt.imshow(mask)
    # plt.show()
    img = cv2.add(img, np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8), mask=mask)
    # plt.imshow(img)
    # plt.show()
    # my_cv_imwrite(save_path + 'point.jpg', img)
    return img


def get_origin_data(data_path):
    big_kinds = ['1肠胃', '2肺病', '3妇科', '4肝胆系统疾病', '5男科', '6内分泌疾病', '7神经系统', '8五官',
                 '9心脑血管疾病', '10其他小类']
    small_kinds = []
    for i in range(len(big_kinds)):
        big_disease_path = data_path + big_kinds[i] + '\\'
        small_kinds.append(os.listdir(big_disease_path))
    return big_kinds, small_kinds


def get_origin_path(origin_data_path, big_kinds, small_kinds, head, img_kind):
    '''
    :param origin_data_path: 原图的路径
    :param big_kinds: 大类标签
    :param small_kinds: 小类标签
    :param head: 命名中第一部分
    :return: 原图的路径+命名
    '''
    path = origin_data_path + big_kinds[head[1][1] - 1] + '\\' + small_kinds[head[1][1] - 1][
        head[1][2] - 1] + '\\' + head[0] + img_kind
    return path

def get_crop(img, pad, area):
    '''
    :param img: 输入图像
    :param pad: 填充参数
    :param area: 命名第三部分，即手掌区域检测结果
    :return: 按照手掌区域检测结果进行剪切后的图像
    '''
    pad_x, pad_y = pad

    x1 = area[0]
    y1 = area[1]
    x2 = area[2]
    y2 = area[3]

    y1 = int(y1 - pad_y) if y1 - pad_y > 0 else 0
    x1 = int(x1 - pad_x) if x1 - pad_x > 0 else 0
    y2 = int(y2 + pad_y) if y2 + pad_y < img.shape[0] else img.shape[0]
    x2 = int(x2 + pad_x) if x2 + pad_x < img.shape[1] else img.shape[1]
    img = img[y1:y2, x1:x2, :]
    # my_cv_imwrite(save_path + 'crop_one.jpg', img)
    return img



def change_KeyPoints(img, KeyPoints):
    '''
    :param img: 原图crop后
    :param KeyPoints: 在192*192下的关键点坐标，此时为float
    :return: 在原图crop后的关键点坐标，为int
    '''
    for i in range(12):
        KeyPoints[i][0] = int(KeyPoints[i][0] / 192 * img.shape[0])
        KeyPoints[i][1] = int(KeyPoints[i][1] / 192 * img.shape[1])
    #     cv2.circle(img, (KeyPoints[i][1], KeyPoints[i][0]), 1, (0, 0, 255), 100)
    # plt.imshow(img)
    # plt.show()
    return KeyPoints


def change_KeyPoints_2(img, KeyPoints):
    '''
    因为第一次剪切是用手掌区域检测的坐标，用关键点置黑后，范围缩小，四周有黑条，删掉可以增加手掌区域面积
    :param img:输入图像
    :param KeyPoints:关键点
    :return:输出图像，关键点
    '''
    left, up, right, down = cal_xyxy(KeyPoints)
    if left < 0:
        left = 0
    if up < 0:
        up = 0
    if right > img.shape[0]:
        right = img.shape[0]
    if down > img.shape[1]:
        down = img.shape[1]
    for i in range(12):
        KeyPoints[i][0] = KeyPoints[i][0] - up
        KeyPoints[i][1] = KeyPoints[i][1] - left
    img = img[up:down, left:right, :]
    # my_cv_imwrite(save_path + 'crop_two.jpg', img)
    return img, KeyPoints


def cal_xyxy(KeyPoints):
    '''
    :param KeyPoints: 12个关键点
    :return: 左上右下4个边界
    '''
    up = min(KeyPoints, key=lambda x: x[0])[0]
    down = max(KeyPoints, key=lambda x: x[0])[0]
    left = min(KeyPoints, key=lambda x: x[1])[1]
    right = max(KeyPoints, key=lambda x: x[1])[1]
    return left, up, right, down


def correction():
    '''
    1、用head把获取原图路径，读取原图
    2、把原图通过area用padding为100进行crop
    3、关键点定位的结果为该crop结果resize到192的结果，转换到该crop结果上
    4、通过转换后的关键点进行背景置黑和矫正
    '''
    # data_path = 'D:\\pycharm\\jiaozheng\\pred_save2\\'
    data_path = 'D:\\pycharm\\jiaozheng\\test\\'
    data_list = os.listdir(data_path)

    origin_data_path = 'E:\\Palam_Data_AfterClear\\'
    big_kinds, small_kinds = get_origin_data(origin_data_path)

    # data_list.sort(key=lambda x: int(x[3:4]))
    data_len = len(data_list)
    save_path = 'D:\\pycharm\\jiaozheng\\test\\'

    # train:val=9:1
    for i in range(int(data_len)):
        name = data_list[i]
        name_after = name_process(name)
        origin_data_name = get_origin_path(origin_data_path, big_kinds, small_kinds, name_after['1'],
                                           name_after['3'][0][-4:])
        try:
            img = my_cv_imread(origin_data_name)
            img = get_crop(img, 100, name_after['3'][1])
            # plt.imshow(img)
            # plt.show()
            KeyPoints = change_KeyPoints(img, name_after['2'][1])
            img, KeyPoints = change_KeyPoints_2(img, KeyPoints)
            img, KeyPoints = Left_or_Right(img, name_after['2'][1])
            img = draw_polygan(img, KeyPoints)
            # plt.imshow(img)
            # plt.show()
            img = warp_affine(img, line12_78(img, KeyPoints))
            img = cv2.resize(img, (512, 512))
            # plt.imshow(img)
            # plt.show()
            my_cv_imwrite(save_path + name, img)
        except Exception as e:
            print(e)
            print(name)


def correction_(img, save_path, landmarks, area,padding):
    '''
    1、用head把获取原图路径，读取原图
    2、把原图通过area用padding为100进行crop
    3、关键点定位的结果为该crop结果resize到192的结果，转换到该crop结果上
    4、通过转换后的关键点进行背景置黑和矫正
    '''

    img = get_crop(img, padding, area)
    KeyPoints = change_KeyPoints(img, landmarks)
    img, KeyPoints = change_KeyPoints_2(img, KeyPoints)
    img, KeyPoints = Left_or_Right(img, landmarks)
    img = draw_polygan(img, KeyPoints)
    # plt.imshow(img)
    # plt.show()
    img = warp_affine(img, line12_78(img, KeyPoints))
    img = cv2.resize(img, (512, 512))
    my_cv_imwrite(save_path + '//' + 'correction.jpg', img)
    return img

if __name__ == '__main__':
    correction()
