# 手掌定位与分类
## 手掌定位
使用resnet18作为backbone,输入为在原图上关键点所在矩形框padding 100px之后crop 并resize到192的图

## 代码说明
训练代码：./train_net/my_train.py(数据读取也在这里)
测试代码：test_image.py

## 手掌分类
在classification中


![avatar](myplot.png)

![avatar](myplot1.png)