import os
from skimage import io
import cv2

import numpy as np

def read_tif(imgpath,bgr_savepath_name):
    img = io.imread(imgpath)  # 读取图片 imgpath为图片所在位置
    img = img / img.max()
    img = img * 255 - 0.001  # 减去0.001防止变成负整型
    img = img.astype(np.uint8)
    print(img.shape)  # 显示图片大小和通道数  通道数为4
    b = img[:, :, 0]  # 蓝通道
    g = img[:, :, 1]  # 绿通道
    r = img[:, :, 2]  # 红通道
    bgr = cv2.merge([b, g, r]) #通道拼接
    cv2.imwrite(bgr_savepath_name, bgr)  # 保存图片
    print(bgr_savepath_name)
def batch_processing(file_path, bgr_savepath,pathname2):
    filelist = os.listdir(file_path)  # 获取当前路径下的文件列表
    i = 0
    for name in filelist:  # 遍历列表下的文件名，其中name与filelist自动对应

        file_path_name = file_path + "\\" + name  # 源文件路径

        #nir_savepath_name = nir_savepath + '/' + '0' + str(i) + '.jpg'  # NIR图像存储路径
        if not os.path.exists(bgr_savepath):
            os.makedirs(bgr_savepath)
        #print(file_path_name)  # 输出文件名进行反馈操作
        bgr_savepath_name = bgr_savepath + "\\" + pathname2 + str(i) + '.jpg'  # BGR图像存储路径
        read_tif(file_path_name,bgr_savepath_name)  # 图像转换
        i += 1
def getfile(rootpath,savebgrpath):
    dir_list = os.listdir(rootpath)
    for pathname in dir_list:
        dir_path = rootpath + "\\"+ pathname
        imagespath = savebgrpath + "\\"+pathname
        batch_processing(dir_path,imagespath,pathname)
if __name__=="__main__":
    readpath = r'E:\遥感数据集\UCMerced_LandUse\Images'
    #readpath = r'‪E:\遥感数据集\UCMerced_LandUse\Images'
    savepath = r'E:\CNNdata\dataset2'
    #savepath = r'E:\CNNdata\dataset2'

    getfile(readpath,savepath)






