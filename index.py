# -*- coding: UTF-8 -*-
import numpy as np
import os
import gzip
import struct
import logging
import mxnet as mx
import matplotlib.pyplot as plt
print("你好世界")
logging.getLogger().setLevel(logging.DEBUG)

#定义获取数据的方法
def read_data(lable_url, image_url): #读入训练数据
    with gzip.open(lable_url) as flb1: # 打开标签文件
        magic, num = struct.unpack(">II", flb1.read(8)) # 读入标签文件头
        label = np.fromstring(flb1.read(), dtype=np.int8) # 读取标签内容
    with gzip.open(image_url) as fimg:
        magic, num, rows, clos = struct.unpack(">IIII", fimg.read(16)) #读取图片文件头，rows 和cols
        image = np.fromstring(fimg.read(), dtype=uint8)
        image = image.reshape(len(label), 1, rows, cols) # 设置正确的数组格式
        image = image.astype(np.float32)/255.0 # 归一化为0 - 1区间
    return (label, image)
    

