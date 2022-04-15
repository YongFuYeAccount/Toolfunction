# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
from os import listdir
import pandas as pd
import time
import math
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader


class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def Normalize(group):
    import numpy as np
    # 公式：newValue = (oldValue-min)/(max)

    minVals = group.min()  # 为0时：求每列的最小值[0 3 1]   .shape=(3,)
    maxVals = group.max()  # 为0时：求每列的最大值[2 7 8]   .shape=(3,)
    ranges = maxVals - minVals

    m = group.shape[0]
    # normDataSet = np.zeros(np.shape(group))  # np.shape(group) 返回一个和group一样大小的数组，但元素都为0
    diffnormData = group - np.tile(minVals, (m, 1))  # (oldValue-min)  减去最小值
    normDataSet1 = diffnormData / np.tile(ranges, (m, 1))

    # print(minVals)  # 打印最小值 [0 3 1]
    # print(maxVals)  # 打印最大值 [2 7 8]
    # print(normDataSet1)
    return normDataSet1

def getlist(Filename):
    FileList = listdir(Filename)
    return FileList

def getimg(video_src_src_path):
    # video_src_src_path = r'C:\Users\86182\Desktop\Dataset\KTH'
    label_name = os.listdir(video_src_src_path)#一级目录下的文件夹名为标签
    label_dir = {}
    index = 0

    frame_save_path = []
    for i in label_name:
        if i.startswith('.'):
            continue
        label_dir[i] = index
        index += 1
        video_src_path = os.path.join(video_src_src_path, i)# #某一类动作视频的总文件夹r'C:\Users\86182\Desktop\Dataset\KTH\boxing'
        video_save_path = os.path.join(video_src_src_path, i) + '_jpg'#某一类动作转化为图片的总文件夹C:\Users\86182\Desktop\Dataset\KTH\boxing_jpg
        if not os.path.exists(video_save_path):
            os.mkdir(video_save_path)#如果不存在与boxing同级的文件夹就创建

        videos = os.listdir(video_src_path)#读取boxing文件夹下的所有文件名
        # 过滤出avi文件
        videos = filter(lambda x: x.endswith('avi'), videos)

        frame_save_path.append(video_save_path)#得到所有——jpg文件夹的list
        for each_video in videos:

            each_video_name, _ = each_video.split('.')#得到每个视频的前缀
            if not os.path.exists(video_save_path + '/' + each_video_name):#在图片的文件夹下面创建以每个视频名前缀为命名的文件夹
                os.mkdir(video_save_path + '/' + each_video_name)

            each_video_save_full_path = os.path.join(video_save_path, each_video_name) + '/'

            each_video_full_path = os.path.join(video_src_path, each_video)

            cap = cv2.VideoCapture(each_video_full_path)
            frame_count = 1
            success = True
            while success:
                success, frame = cap.read()
                # print('read a new frame:', success)

                params = []
                params.append(1)
                if success:
                    cv2.imwrite(each_video_save_full_path + each_video_name + "_%d.jpg" % frame_count, frame, params)

                frame_count += 1
            cap.release()
    # np.save('kth_label_dir.npy', label_dir)
    # print(label_dir)
    return frame_save_path
#定义读取文件夹里文件名的文件函数
def getlist(Filename):
    FileList = listdir(Filename)
    return FileList

##定义拼接元素的函数##
def splicedata(List1, element):
    for i in range(len(List1)):
        List1[i] = os.path.join(element, List1[i])
    return List1

#####一级文件data里面包含了run,walk,wave,clap四个子文件夹，run里面包含了具体的avi数据########
###输入的文件夹路径为data路径###
fir_grade_filepath = input('请输入目标测试数据的文件夹路径：')#C:\Users\86182\Desktop\Spring\KTH-HMDB51\Show\data1
sec_grade_filepath = getlist(fir_grade_filepath)#['clap', 'run']


img_src_path = getimg(fir_grade_filepath)#得到了每个视频的每一帧的图片，并放回存放视频帧的文件夹
#['C:\\Users\\86182\\Desktop\\Spring\\KTH-HMDB51\\Show\\data1\\clap_jpg', 'C:\\Users\\86182\\Desktop\\Spring\\KTH-HMDB51\\Show\\data1\\run_jpg']


path_data_dict = dict()#创建空字典，计划用地址与数据对应
label_data_dict = dict()#创建空字典，计划用标签与数据对应
for i in range(len(img_src_path)):

    each_avi_img_path = getlist(img_src_path[i])#取出每一类中的图片#['#20_Rhythm_clap_u_nm_np1_fr_goo_0']


    for b in range(len(each_avi_img_path)):
        # 获取目录的路径
        img_temp_dir = os.path.join(img_src_path[i],each_avi_img_path[b])

        datalabel = img_temp_dir.split('\\')[9]#取出对应的标签
        # datalabel = datalabel.split('_')[0]
        # print(datalabel)
        # print(img_temp_dir)
        #获取该目录下所有的文件
        img_list = os.listdir(img_temp_dir)#一个视频对应的所有帧文件路径

        #遍历所有的文件名称
        totaldata = []
        for img_name in img_list:
                #判断文件是否为目录,如果为目录则不处理
            if not os.path.isdir(img_name):
                    #获取图片的路径
                img_path = os.path.join(img_temp_dir,img_name)
                    #因为图片是黑白的，所以以灰色读取图片
                img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, (15, 15))
                # print(type(res))#<class 'numpy.ndarray'>

                data = Normalize(res)#对数据进行预处理
                #获取图片的像素

                # print(type(row_data))#<class 'list'>

                # print(row_data)
                totaldata.append(data.flatten())
                #将图片数据写入到csv文件中
                # writer = csv.writer(f)
                # writer.writerow(row_data)
        # print(totaldata)
        # print(type(totaldata))
        # print(len(totaldata))
        data = np.array(totaldata)
        path_data_dict.setdefault(img_temp_dir, data)
        label_data_dict.setdefault(datalabel, data)

# print(label_data_dict,path_data_dict)i
label = []
for i in label_data_dict.keys():#取出键值并将键值存到列表中
    label.append(i)

# for data1 in label_data_dict.values():#取出健对应的值
#     print(data1)


total_data_variable = []
total_label_variable = []
for i in range(len(label)):#生成与健相同个数的变量并将数据存入相对应的变量中#并构建相同数量额标签变量

    a = 'label' + str(i)
    exec(a + '= %r' % [])#批量生成标签变量

    b = 'mat' + str(i)
    exec(b + '= %r' % [])

    b = label_data_dict[label[i]]
    a = label[i]

    total_data_variable.append(b)
    total_label_variable.append(a)

# print(total_data_variable)
# print(total_label_variable)
# print(type(total_label_variable[0]))
# print(len(total_label_variable))
# print(len(total_data_variable))

##########
#构建一个函数能够把得到的数据帧剪切为帧的倍数#
#########

#给定对应标签函数#
def get_label_result(label):

    result0 = 'clap'
    result1 = 'run'
    result2 = 'walk'
    result3 = 'wave'

    if result0 in label:
        output_result = 0
    elif result1 in label:
        output_result = 1
    elif result2 in label:
        output_result = 2
    elif result3 in label:
        output_result = 3

    return output_result



##对数据进行处理##
def cut_data(Frame,data,label):
    total_frame = len(data)
    int_frame = int(total_frame / Frame)
    data = data[0:int_frame*Frame]
    decide_result = get_label_result(label)
    frame_label = [decide_result] * int_frame

    return data ,frame_label

# totalMat = [[]]
# total_Labels = [[]]
#将数据已经读取完毕#
for i in range(len(total_data_variable)):
    total_data_variable[i], total_label_variable[i] = cut_data(20,total_data_variable[i], total_label_variable[i])#目标样本的帧数

    if i==0 :
        totalMat =  total_data_variable[i]  # 垂直组合numpy
        total_Labels =  total_label_variable[i] # 垂直组合numpy
    else:
        totalMat = np.concatenate((totalMat, total_data_variable[i]), axis=0)# 垂直组合numpy
        total_Labels = np.concatenate((total_Labels, total_label_variable[i]), axis=0)# 垂直组合numpy



print('###########')
print(totalMat)
print(total_Labels)

# #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
totaldata = np.array(totalMat)
totaldata = totaldata.reshape((len(total_Labels), 1, 60, 75))#
totallable = total_Labels

print('##########')
print(len(total_Labels))
print(totaldata.shape)

test_x = torch.tensor(totaldata).type(torch.FloatTensor)
test_y = torch.tensor(totallable, dtype=torch.long)  # csv读取int转化为long

BATCH_SIZE = 1
##将标签和输入数据用自定义函数封装##
input_test_data = GetLoader(test_x, test_y)
target_test_data_loader = DataLoader(input_test_data, batch_size=BATCH_SIZE, shuffle=False)

# _ , data_loader = source_loader(16)
# for step, (images, labels) in enumerate(target_test_data_loader):
#     print(images)
#     print(labels)
#     print(type(images))
#     print(images.shape)
#     print(labels.shape)











# def test_data_loader(BATCH_SIZE,Frame,dict_data):
#
#     # 构建pytorch行驶本数据集函数
#     # 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
#     class GetLoader(torch.utils.data.Dataset):
#         # 初始化函数，得到数据
#         def __init__(self, data_root, data_label):
#             self.data = data_root
#             self.label = data_label
#
#         # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
#         def __getitem__(self, index):
#             data = self.data[index]
#             labels = self.label[index]
#             return data, labels
#
#         # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
#         def __len__(self):
#             return len(self.data)
#
#
#     # 训练数据的构建 读取数据集中的所有数据
#     Mat1 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\Init_data\kth\clapping.csv')
#     Mat2 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\Init_data\kth\running.csv')
#     Mat3 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\Init_data\kth\walking.csv')
#     Mat4 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\Init_data\kth\waving.csv')
#     totalMat = np.concatenate((Mat1[0:42660], Mat2[0:38500], Mat3[0:45780], Mat4[0:53660] ), axis=0)  # 垂直组合numpy
#     # trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0) #[42667, 38505, 65795, 53678] 200645
#
#     Labels1 = [0] * 2133#42660    42667
#     Labels2 = [1] * 1925# 38505  38500
#     Labels3 = [2] * 2289 # 65795 657803289
#     Labels4 = [3] * 2683 # 53660     53678   totallabel= 10030
#
#     total_Labels = np.concatenate((Labels1, Labels2, Labels3, Labels4), axis=0)  # 垂直组合numpy
#
#
#
#     # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
#     # totaldata = np.array(totalMat - 18)  # 对数据进行预处理
#
#     totaldata = np.array(totalMat)
#     totaldata = totaldata.reshape((len(total_Labels), 1, 60, 75))#
#     totallable = total_Labels
#
#     # 打乱索引
#     # 得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
#     totaldata = np.array(totaldata)
#     totalLabels = np.array(total_Labels).reshape(len(totallable), )  # 将标签修改成一维
#
#     index = [i for i in range(len(totaldata))]  # test_data为测试数据
#     np.random.shuffle(index)  # 打乱索引
#     totaldata = totaldata[index]
#     totallable = totalLabels[index]
#
#     # 划分训练集和测试集数据80%训练集，20%测试集
#     trainX = totaldata[0:7224]
#     train_Labels = totallable[0:7224]
#
#     testX = totaldata[7224:]#200645
#     test_Labels = totallable[7224:]
#
#     np.save("20Fsource_train_data.npy", trainX)
#     np.save('20Fsource_train_label.npy', train_Labels)
#
#     np.save("20Fsource_test_data.npy", testX)
#     np.save('20Fsource_test_label.npy', test_Labels)
#
#     ##封装训练集数据：
#     # 将nparray 转化为tensor，将其打乱，并重写训练数据集格式为pytorch需要输入的形式
#     train_x = torch.tensor(trainX).type(torch.FloatTensor)
#     train_y = torch.tensor(train_Labels, dtype=torch.long)  # csv读取int转化为long
#
#     # 将标签和输入数据用自定义函数封装
#     input_train_data = GetLoader(train_x, train_y)
#     source_train_data_loader = DataLoader(input_train_data, batch_size=BATCH_SIZE, shuffle=True)
#
#     # 测试集数据
#     test_x = torch.tensor(testX).type(torch.FloatTensor)
#     test_y = torch.tensor(test_Labels, dtype=torch.long)  # csv读取int转化为long
#
#     input_test_data = GetLoader(test_x, test_y)
#     source_test_data_loader = DataLoader(input_test_data, batch_size=BATCH_SIZE, shuffle=True)
#
#     return source_train_data_loader,  source_test_data_loader,test_x,test_y
#

# def convert_img_to_csv(img_dir, savepath):
#     #设置需要保存的csv路径
#     with open(savepath,'w',newline='') as f:
#
#         #该目录下有9个目录,目录名从0-9
#         c = getlist(img_dir)
#
#         # print(c)
#         # print(len(c))
#         for i in range(len(c)):
#             #获取目录的路径
#             img_temp_dir = os.path.join(img_dir,c[i])
#             #获取该目录下所有的文件
#             img_list = os.listdir(img_temp_dir)#1356#
#             # print(type(img_list))
#             # print(type(img_list[0]))
#             # print(img_list)
#             # print(len(img_list))
#             #遍历所有的文件名称
#             for img_name in img_list:
#                 #判断文件是否为目录,如果为目录则不处理
#                 if not os.path.isdir(img_name):
#                     #获取图片的路径
#                     img_path = os.path.join(img_temp_dir,img_name)
#                     #因为图片是黑白的，所以以灰色读取图片
#                     img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
#
#                     ########resize img ########
#                     # res = cv2.resize(img, (100, 100))
#                     # res = cv2.resize(res,(160,120))
#
#                     res = cv2.resize(img,(15,15))
#                     print(type(res)
#                     #图片标签
#                     # row_data = [i]
#                     # row_data = []
#                     # #获取图片的像素
#                     # row_data.extend(res.flatten())
#                     # #将图片数据写入到csv文件中
#                     # writer = csv.writer(f)
#                     # writer.writerow(row_data)
