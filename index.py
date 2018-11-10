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
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16)) #读取图片文件头，rows 和cols
        image = np.fromstring(fimg.read(), dtype=np.uint8)
        image = image.reshape(len(label), 1, rows, cols) # 设置正确的数组格式
        image = image.astype(np.float32)/255.0 # 归一化为0 - 1区间
    return (label, image)
#读入数据
(train_lb1, train_img) = read_data('train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz')
(val_lb1, val_img) = read_data('t10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz')

batch_size = 32; #批大小
#迭代器
train_iter = mx.io.NDArrayIter(train_img, train_lb1, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_img, val_lb1, batch_size)

for i in range(10): # 输出钱10个数字
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i].reshape(28,28), cmap='Greys_r')
    plt.axis('off')
#plt.show() # 显示图像
#print('lable:%s' % (train_lb1[0:10],))

data = mx.symbol.Variable('data')
#将图片摊平，例如28*28的图片会变成784个数据点，这样才可能与普通神经元连接
flatten = mx.sym.Flatten(data=data, name="flatten")
#第一层网络及非线性激活，有128个神经元，使用ReLU非线性
fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=128, name="fc1")
act1 = mx.sym.Activation(data=fc1, act_type="relu", name="act1")

#第二层网络及非线性激活，有64个神经元，使用ReLU非线性激活函数
fc2 = mx.sym.FullyConnected(data=act1, num_hidden=64, name="fc2")
act2 = mx.sym.Activation(data=fc2, act_type="relu", name="act2")

#输出神经元，因为需要分10类，所以有10个神经元
fc3 = mx.sym.FullyConnected(data=act2, num_hidden=10, name="fc3")
#SoftMax层，将上一层输入的10个数变成10个分类的概率
net = mx.sym.SoftmaxOutput(data=fc3, name="softmax")

#显示网络参数情况
shape={"data":(batch_size, 1, 28, 28)}
#用命令行显示网络参数情况
mx.viz.print_summary(symbol=net, shape=shape)
#显示网络结构图
mx.viz.plot_network(symbol=net, shape=shape).view()


#训练网络
module=mx.mod.Module(symbol=net, context=mx.cpu(0))
module.fit(
	train_iter,
    eval_data = val_iter,
    optimizer = 'sgd',
    optimizer_params = {'learning_rate' : 0.2, 'lr_scheduler' : mx.lr_scheduler.FactorScheduler(step=6000/batch_size, factor=0.9)},
    num_epoch = 20,
    batch_end_callback = mx.callback.Speedometer(batch_size, 6000/batch_size)
	
)
