from django.http import HttpResponse
from django.shortcuts import render
import os, sys
from django.http import JsonResponse
destination_path = './pytorch/data'
detection_save_path = './pytorch/data'
import json
from pytorch.fuse_main import start_diagnose
"""
global variable
"""
def wx_upload_file(request):
    print('we are in wx upload file')
    if request.method == "POST":  # 请求方法为POST时，进行处理
        myFile = request.FILES.get("file", None)  # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join(destination_path, myFile.name), 'wb+')  # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():  # 分块写入文件
            destination.write(chunk)
        destination.close()
        description = detection(myFile.name)
        print('The output description3 is:' ,description,type(description))

        res = {'diagnose_result':'http://123.207.223.60:8000/result/'+myFile.name,
               'diagnose_yolo':'http://123.207.223.60:8000/mid_result/'+myFile.name+'/resize_192.jpg',
               'diagnose_crop':'http://123.207.223.60:8000/mid_result/'+myFile.name+'/crop_one.jpg',
               'diagnose_align':'http://123.207.223.60:8000/mid_result/'+myFile.name+'/correction.jpg',
               'diagnose_words':description}
        return JsonResponse(res, json_dumps_params={'ensure_ascii': False})

def upload_file(request):
    print(request)
    print('start to run the function upload_file')
    if request.method == "POST":    # 请求方法为POST时，进行处理
        myFile =request.FILES.get("myfile", None)    # 获取上传的文件，如果没有文件，则默认为None
        if not myFile:
            return HttpResponse("no files for upload!")
        destination = open(os.path.join(destination_path,myFile.name),'wb+')    # 打开特定的文件进行二进制的写操作
        for chunk in myFile.chunks():      # 分块写入文件
            destination.write(chunk)
        destination.close()
        detection(myFile.name)

        # res = {'score': '1', 'input_img_name': myFile.name, 'diagnose_result': myFile.name}
        # return JsonResponse(res, json_dumps_params={'ensure_ascii': False})
        return HttpResponse('http://127.0.0.1:8000/result/'+myFile.name)
def upload_page(request):
    print('start to run uplaod_page function')
    return render(request, 'uploadPage.html')


def detection(img_name):
    print('start to run press_detection_button function')
    # running forward process
    print('开始进入testEnterFunction')
    print(img_name)
    description = start_diagnose(img_name=img_name)
    return description