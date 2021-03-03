import sys,os
sys.path.append('pytorch')
import Palm_correction as four
import Palm_key_point_positioning as three
import make_data as second
import Palm_area_detection as first
import argparse
import cv2
import os
import torch
import numpy as np

from Palm_loc_and_cls.classification.test_child import Diagnose
sys.path.append('./Palm_loc_and_cls/classification')
def my_resize(img, save_path, origin_save_path):
    if img.shape[0] < img.shape[1]:  # h<w
        # cv2.circle(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), 4, (0, 255, 255), 100)
        rotate = cv2.getRotationMatrix2D((img.shape[0] * 0.5, img.shape[0] * 0.5), -90, 1)
        origin_img = cv2.warpAffine(img, rotate, (img.shape[0], img.shape[1]))
        # my_cv_imwrite(origin_save_path, origin_img)
        # print('true')
    else:
        origin_img = img
    img = cv2.resize(origin_img, (192, 192))
    my_cv_imwrite(save_path + '//' + 'resize_192.jpg', img)
    return origin_img, img


def my_cv_imread(filepath):
    # 使用imdecode函数进行读取
    img = cv2.imdecode(np.fromfile(filepath, dtype=np.uint8), -1)
    return img


def my_cv_imwrite(filepath, img):
    # 使用imencode函数进行读取
    cv2.imencode('.jpg', img)[1].tofile(filepath)


def crop_one(img, area, save_path, pad):
    x1, y1, x2, y2 = area
    pad_x, pad_y = pad
    x1 = int(img.shape[1] * x1)
    y1 = int(img.shape[0] * y1)
    x2 = int(img.shape[1] * x2)
    y2 = int(img.shape[0] * y2)

    y1 = int(y1 - pad_y) if y1 - pad_y > 0 else 0
    x1 = int(x1 - pad_x) if x1 - pad_x > 0 else 0
    y2 = int(y2 + pad_y) if y2 + pad_y < img.shape[0] else img.shape[0]
    x2 = int(x2 + pad_x) if x2 + pad_x < img.shape[1] else img.shape[1]
    img = img[y1:y2, x1:x2, :]

    img = cv2.resize(img, (192, 192))
    my_cv_imwrite(save_path + '//' + 'crop_one.jpg', img)
    return img, [x1, y1, x2, y2]



def get_crop(img, pad, area):
    '''
    :param img: 输入图像
    :param pad: 填充参数
    :param area: 命名第三部分，即手掌区域检测结果
    :return: 按照手掌区域检测结果进行剪切后的图像
    '''
    x1 = area[0]
    y1 = area[1]
    x2 = area[2]
    y2 = area[3]
    img = img[y1 - pad:y2 + pad, x1 - pad:x2 + pad, :]
    # my_cv_imwrite(save_path + 'crop_one.jpg', img)
    return img

def advance_judge(img):
    '''
    3024*4032=(75,100)
    897*1920=(25,50)
    判断手掌的分辨率是否合格，并且确定padding大小
    :param img:
    :return:
    '''
    if img.shape[0] < 1000 and img.shape[1] < 1000:
        print("图片分辨率过低，请重新拍摄")
        sys.exit()
    else:
        pad_x = img.shape[0] / 40
        pad_y = img.shape[1] / 40
        # print(pad_x, pad_y)

    return (100,100)
    # return (pad_x, pad_y)

def start_diagnose(img_name):
    '''
    zero把图片resize到192
    first用于手掌区域检测，输入192*192的图，输出标记框的图和label.txt
    second用于把检测结果的xywh转换为xyxy，输入xywh的label.txt，输出xyxy的label.txt
    three用于关键点定位，输入原图中根据label.txt中坐标padding100个像素之后再resize到192的图，输出12个关键点
    four用于根据关键点定位后的结果进行矫正，输入结果图、原图，输出512*512的矫正图
    '''
    parser = argparse.ArgumentParser()


    parser.add_argument('--name', default='hy.jpg', help='name')
    parser.add_argument('--origin_source', type=str, default='pytorch/data', help='source')
    parser.add_argument('--source', type=str, default='', help='source')
    parser.add_argument('--weight_one', nargs='+', type=str, default='pytorch/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=192, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    parser.add_argument('--weight_two', nargs='+', type=str, default='pytorch/min_palm_model_epoch.pt', help='model.pt path(s)')
    parser.add_argument('--weight_three', nargs='+', type=str, default='pytorch/child_cls_201.pt',
                        help='model.pt path(s)')

    opt = parser.parse_args([])

    opt.name = img_name
    origin_img = my_cv_imread(os.path.abspath(opt.origin_source) +'' + '//' + opt.name)
    opt.source = os.path.abspath(opt.project) + '//' + opt.name
    if not os.path.exists(opt.source):
        os.makedirs(opt.source)

    origin_img, img = my_resize(origin_img, opt.source, opt.origin_source + '//' + opt.name)
    padding = advance_judge(origin_img)
    txt_path = first.detect(opt)
    area = second.xywh2xyxy_(txt_path)
    img, area = crop_one(origin_img, area, opt.source, padding)

    test = three.PalmTest(image_dir=opt.source, o_net_path=opt.weight_two)
    landmarks = [[float(test.landmarks[2 * i]), float(test.landmarks[2 * i + 1])] for i in range(12)]
    img = four.correction_(origin_img, opt.source, landmarks, area,padding)

    description = Diagnose(img=img,model_path=opt.weight_three,output_path='pytorch/diagnose_result',img_name=opt.name).start()
    print('The output description2 is :',description)
    return description

if __name__ == '__main__':
    # '''
    # zero把图片resize到192
    # first用于手掌区域检测，输入192*192的图，输出标记框的图和label.txt
    # second用于把检测结果的xywh转换为xyxy，输入xywh的label.txt，输出xyxy的label.txt
    # three用于关键点定位，输入原图中根据label.txt中坐标padding100个像素之后再resize到192的图，输出12个关键点
    # four用于根据关键点定位后的结果进行矫正，输入结果图、原图，输出512*512的矫正图
    # '''
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--name', default='chr.jpg', help='name')
    # parser.add_argument('--origin_source', type=str, default='data', help='source')
    # parser.add_argument('--source', type=str, default='', help='source')
    # parser.add_argument('--weight_one', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    # parser.add_argument('--img-size', type=int, default=192, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--view-img', action='store_true', help='display results')
    # parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    # parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--update', action='store_true', help='update all models')
    # parser.add_argument('--project', default='runs', help='save results to project/name')
    # parser.add_argument('--exist-ok', action='store_false', help='existing project/name ok, do not increment')
    # parser.add_argument('--weight_two', nargs='+', type=str, default='min_palm_model_epoch.pt', help='model.pt path(s)')
    # opt = parser.parse_args()
    #
    # origin_img = my_cv_imread(os.path.abspath(opt.origin_source) + '//' + opt.name)
    # opt.source = os.path.abspath(opt.project) + '//' + opt.name
    # if not os.path.exists(opt.source):
    #     os.makedirs(opt.source)
    #
    # origin_img, img = my_resize(origin_img, opt.source, opt.origin_source + '//' + opt.name)
    # txt_path = first.detect(opt)
    # area = second.xywh2xyxy_(txt_path)
    # img, area = crop_one(origin_img, area, opt.source, 100)
    #
    # test = three.PalmTest(image_dir=opt.source, o_net_path=opt.weight_two)
    # landmarks = [[float(test.landmarks[2 * i]), float(test.landmarks[2 * i + 1])] for i in range(12)]
    # img = four.correction_(origin_img, opt.source, landmarks, area)
    # Diagnose(img=img,model_path='child_cls_201.pt',output_path='./')
    start_diagnose('hy.jpg')