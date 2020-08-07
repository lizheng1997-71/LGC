# import data
# import argparse
# from LGCmodel import LGC
# parser = argparse.ArgumentParser()
# parser.add_argument("--dataset",default="E:\\LGC\\dataset")
# parser.add_argument("--imgsize",default=41,type=int)
# parser.add_argument("--scale",default=2,type=int)
#
# parser.add_argument("--output_channels",default=1,type=int)
# parser.add_argument("--batchsize",default=128,type=int)
# parser.add_argument("--savedir",default="‪E:\\LGC\\model")
# parser.add_argument("--iterations",default=100,type=int)
#
# args = parser.parse_args()
# data.load_dataset(args.dataset)
# # if args.imgsize % args.scale != 0:
# #     print(f"Image size {args.imgsize} is not evenly divisible by scale {arg.scale}")
# #     exit()
# up_size = args.imgsize*args.scale
# network = LGC(args.imgsize,args.scale,args.output_channels)
# network.set_data_fn(data.get_batch(args.imgsize,args.scale,args.batchsize))
# network.train(args.iterations,args.savedir)
# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import math
# import cv2
# def BiBubic(x):
#     x=abs(x)
#     if x<=1:
#         return 1-2*(x**2)+(x**3)
#     elif x<2:
#         return 4-8*x+5*(x**2)-(x**3)
#     else:
#         return 0
#
# def BiCubic_interpolation(img,dstH,dstW):
#     scrH,scrW,_=img.shape
#     #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
#     retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
#     for i in range(dstH):
#         for j in range(dstW):
#             scrx=i*(scrH/dstH)
#             scry=j*(scrW/dstW)
#             x=math.floor(scrx)
#             y=math.floor(scry)
#             u=scrx-x
#             v=scry-y
#             tmp=0
#             for ii in range(-1,2):
#                 for jj in range(-1,2):
#                     if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
#                         continue
#                     tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
#             retimg[i,j]=np.clip(tmp,0,255)
#     return retimg
# im_path='E:\LGC\dataset\\train\\agricultural\\agricultural1.jpg'
# image=np.array(Image.open(im_path))
# image3=BiCubic_interpolation(image,image.shape[0]*2,image.shape[1]*2)
# image3=Image.fromarray(image3.astype('uint8')).convert('RGB')
# cv2.imshow('ima', image3)
import tensorflow as tf
# import tensorflow_datasets as tfs
from data import load_dataset,get_dataset,load_preprosess_image
dataset_dir = 'E:\LGC\dataset'
from LGCmodel import LGC

down_size = 41
scale = 2
trainset = load_dataset(dataset_dir)
train_image,labels = get_dataset(down_size,scale)




import matplotlib.pyplot as plt



train_dataset = tf.data.Dataset.from_tensor_slices((train_image, labels))  # 用load_preprosess_image对图片做一个读取预处理 速度有些慢

AUTOTUNE = tf.data.experimental.AUTOTUNE  # 根据计算机cpu的个数自动的做并行运算  临时实验方法 有可能变化

train_dataset = train_dataset.map(load_preprosess_image,
                                  num_parallel_calls=AUTOTUNE)  # .map是使函数应用在load_preprosess_image中所有的图像上
num_epoch = 80
BATCH_SIZE = 128
learning_rate = 0.01

train_count = len(train_image)

train_dataset = train_dataset.shuffle(train_count).batch(BATCH_SIZE)

train_dataset = train_dataset.prefetch(AUTOTUNE)  # 前台在训练时 后台读取数据 自动分配cpu

#imgs, las = next(iter(train_dataset))  # 取出的是一个batch个数的图片 shape = (batch_size,256,256,3)




# dataset = tfds.load("tf_flowers", split=tfds.Split.TRAIN, as_supervised=True)
# dataset = dataset.map(lambda img, label: (tf.image.resize(img, (224, 224)) / 255.0, label)).shuffle(1024).batch(batch_size)
model = LGC()
optimizer = tf.keras.optimizers.SGD(lr=learning_rate,momentum=0.9)
for e in range(num_epoch):
    for images, labels in train_dataset:
        with tf.GradientTape() as tape:
            labels_pred = model(images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=labels, y_pred=labels_pred)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    print("loss %f" % loss.numpy())