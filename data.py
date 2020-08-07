
# import random
import numpy as np
import os
import imageio
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
# import scipy.misc
# img = imageio.imread(myImage)替代scipy.misc。imread
from PIL import Image
# img = np.array(Image.fromarray(myImage).resize((num_px,num_px))替代scipy.misc.imresize
train_set = []
train_labels = []
# val_set = []
# test_set = []
# batch_index = 0

"""
Load set of images in a directory.
This will automatically allocate a 
random 20% of the images as a test set
data_dir: path to directory containing images
"""


def load_dataset(data_dir):
    train_data_dir = data_dir + "\\" + "train"
    # val_data_dir = data_dir + "\\" + "Val"
    # test_data_dir = data_dir + "\\" + "test"
    train_img_files = os.listdir(train_data_dir)
    # Val_img_files = os.listdir(val_data_dir)
    # test_img_files = os.listdir(test_data_dir)
    # ori_size = down_size*2
    a = 0
    b = 0
    for a,dir_train in enumerate(train_img_files):
        imagesdir = os.listdir(os.path.join(train_data_dir,dir_train))
        for b,i in enumerate(imagesdir):
        #img = scipy.misc.imread(data_dir+img_files[i])

            image1 = os.path.join(train_data_dir,dir_train,i)
            # img = cv2.imread(image1)
            # shrunkimg = tf.image.resize(img, (shrunk_size, shrunk_size), method=2)
            obg = cv2.imread(image1)
            # imagepatch = cv2.resize(obg,None,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
            # cv2.imshow("image1",image1)
            # imagepatch = crop(obg,ori_size)

            train_set.append(obg)
            b += 1
        a += 1
            # train_labels.append(shrunkimg)
        # if i in test_indices:
        #     test_set.append(data_dir+"/"+img_files[i])
        # else:
    # for dir_Val in Val_img_files:
    #     Valdir = os.listdir(val_data_dir + "\\" +dir_Val)
    #     for a in Valdir:
    #         image2 = cv2.imread(train_data_dir + "\\" + dir_Val + a)
    #         val_set.append(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY))
    # for dir_test in test_img_files:
    #
    #     test_images_dir = os.listdir(test_data_dir + "\\" +dir_test)
    #     for b in test_images_dir:
    #
    #         image3 = cv2.imread(test_data_dir + "\\" + dir_test + b)
    #         test_set.append(cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY))



    return train_set
           # val_set , test_set






def get_dataset(down_size,scale):

    x_batch = []
    y_batch = []
    imagepatchs = []
    original_size = down_size*scale
    # traindata = get_imagepatch(get_shrunkimg(train_set,shrunk_size),)
    for image in train_set:
        imagepatchs.append(crop(image,original_size))
        for q,imgpatch in enumerate(imagepatchs):
            for p,simg in enumerate(imgpatch):
                y_batch.append(simg)
    # y_batch = [i for j in y_batch for i in j]
                x_batch.append(get_shrunkimg(simg,scale))
    # max_counter = len(train_labels) / batch_size

    return x_batch, y_batch






# def get_imagepatch(dataset,size):
#     crop_data = []
#     for image in dataset:
#
#         # for image in imagelist:
#             crop_data.append(crop(image,size))
#             # t += 1
#
#
#         # i += 1
#     return crop_data
def get_shrunkimg(img1,scale):
    # shimgs = []
    # for p,imglist in enumerate(data):
    #
    #     for q,img1 in enumerate(imglist):

            # image_data = tf.image.decode_jpeg(img)
            # img = np.int8(img)
            # cv_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # image_array = np.array(img.convert("RGB"))
            # shrunkimg = cv2.resize(cv_img, (shrunk_size,shrunk_size), interpolation=cv2.INTER_CUBIC)
            # image_array = Image.fromarray(np.uint8(img1), mode='RGB')
            # image_array = Image.fromarray(img1.astype('uint8')).convert('RGB')
            # image_array = np.asanyarray(img1, dtype=np.uint8)
            # image_array = Image.fromarray(np.uint8(image_array), mode='RGB')
    shrunkimg = cv2.resize(img1,None,None,fx=1/scale,fy=1/scale,interpolation=cv2.INTER_CUBIC)
                # Image.resize(image_array, (shrunk_size, shrunk_size), Image.BICUBIC)
            # plt.show(shrunkimg)
            # shrunkimg = Image.fromarray(img).resize((shrunk_size, shrunk_size)
        #     shimgs.append(shrunkimg)
        #     q += 1
        # p += 1
    return shrunkimg





def crop (image00,crop_size):
    x = image00.shape[0]
    y = image00.shape[1]
    crop_num1 = x // crop_size
    crop_num2 = y // crop_size
    #image_arr = np.array(image00)
    a = 0
    b = 0
    imagepatch = []
    # imagepatchs = []
    for a in range(crop_num1):

        for b in range(crop_num2):

            cutimg = image00[a*crop_size:(a+1)*crop_size,b*crop_size:(b+1)*crop_size]
            # plt.show(cutimg)
            imagepatch.append(cutimg)
            b += 1
        a += 1

    return imagepatch
def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
    # cv2.imshow("image1", cv_img)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    return cv_img
def load_preprosess_image(input, label):
    # image = tf.io.read_file(input)  # 读取的是二进制格式 需要进行解码
    # image = tf.image.decode_jpeg(image, channels=3)  # 解码 是通道数为3
    image = cv2.resize(input, None, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_CUBIC)
    # image = .image.resize(input, [256, 256])  # 统一图片大小
    image = tf.cast(image, tf.float32)  # 转换类型
    image = image / 255  # 归一化

    # label = tf.io.read_file(label)
    # label = tf.image.decode_jpeg(label, channels=3)  # 解码 是通道数为3
    # label = tf.image.resize(label, [256, 256])  # 统一图片大小
    label = tf.cast(label, tf.float32)  # 转换类型
    label = label / 255  # 归一化

    return image, label  # return回的都是一个batch一个batch的 ， 一个批次很多张tf
