import os
import random

from shutil import copy2
def getDir(filepath):
    pathlist = os.listdir(filepath)
    return pathlist
def mkTotalDir(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dic = ['train','Val','test']
    for i in range(0,3):
        current_path=data_path+dic[i]+'/'
        #这个函数用来判断当前路径是否存在，如果存在则创建失败，如果不存在则可以成功创建
        isExists = os.path.exists(current_path)
        if not isExists:
            os.makedirs(current_path)
            print('successful '+dic[i])
        else:
            print('is existed')
    return
def getclassesMes(source_path):
    classnamelist = getDir(source_path)
    classnum = len(classnamelist)
    return classnamelist,classnum
def mkClassDir(source_path,change_path):
    classnamelist,classnum = getclassesMes(source_path)
    for i in range(0,classnum):
        current_class_path = os.path.join(change_path,classnamelist[i])
        isExist = os.path.exists(current_class_path)
        if not isExist:
            os.makedirs(current_class_path)
            print('successful'+ classnamelist[i])
        else:
            print('is existed')
"""
source_path:原始多类图像的存放路径
train_path:训练集图像的存放路径

test_path:测试集图像的存放路径
"""
def divideTrainTest(source_path,train_path,Val_path,test_path):
    classes_name_list,class_num = getclassesMes(source_path)
    mkClassDir(source_path,train_path)
    mkClassDir(source_path,Val_path)
    mkClassDir(source_path,test_path)
    for i in range(0,class_num):
        source_image_dir = os.listdir(source_path + "\\" + classes_name_list[i] + "\\")
        random.shuffle(source_image_dir)
        train_image_list = source_image_dir[0:int(0.4*len(source_image_dir))]
        Val_image_list = source_image_dir[int(0.4*len(source_image_dir)):int(0.5*len(source_image_dir))]
        test_image_list = source_image_dir[int(0.5*len(source_image_dir)):]
        for train_image in train_image_list:
            origins_train_image_path = source_path + "\\" + classes_name_list[i] + "\\" + train_image
            new_train_image_path = train_path + "\\" + classes_name_list[i] + "\\"
            copy2(origins_train_image_path, new_train_image_path)
        for validation_image in Val_image_list:
            origins_validation_image_path = source_path +"\\" + classes_name_list[i] + "\\" + validation_image
            new_validation_image_path = Val_path + "\\" + classes_name_list[i] + "\\"
            copy2(origins_validation_image_path, new_validation_image_path)

        for test_image in test_image_list:
            origins_test_image_path = source_path + "\\" + classes_name_list[i] + "\\" + test_image
            new_test_image_path = test_path + "\\" + classes_name_list[i] + "\\"
            copy2(origins_test_image_path, new_test_image_path)

if __name__=="__main__":
    data_path = r'E:\LGC\数据集'
    source_path =r'E:\CNNdata\dataset2'
    train_path = r'E:\LGC\数据集\train'
    Val_path = r'E:\LGC\数据集\Val'
    test_path = r'E:\LGC\数据集\test'
    mkTotalDir(data_path)
    divideTrainTest(source_path,train_path,Val_path,test_path)