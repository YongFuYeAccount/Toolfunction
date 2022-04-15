
import csv
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

def target_loader(BATCH_SIZE):
    # 读csv函数(读数据)
    def R_xcsv(Filedirectory):
        csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        date = []  # 创建列表准备接收csv各行数据
        renshu = 0
        for one_line in csv_reader_lines:
            date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
            renshu = renshu + 1  # 统计行数

        # 读取的date是字符串，将其转化为数值
        for i in range(len(date)):
            date[i] = list(map(float, date[i]))
        # for i in range(len(date)):
        #     date[i] = np.array(date[i]).reshape(8, 8)#将列表的元素转化为8 x 8
        # date = np.array(date, dtype=float)  # trainX为待转化的列表
        return date  # 返回的数据是浮点型存在list中

    # 读标签函数
    def R_ycsv(Filedirectory):
        csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
        csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
        date = []  # 创建列表准备接收csv各行数据
        renshu = 0
        for one_line in csv_reader_lines:
            date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
            renshu = renshu + 1  # 统计行数

        for i in range(len(date)):  # 将读取的字符串转化为数值
            date[i] = list(map(int, date[i]))
        # date = np.array(date, dtype=float)  # trainX为待转化的列表
        return date

    # 构建pytorch行驶本数据集函数
    # 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
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

    # 训练数据的构建 读取数据集中的所有数据
    Mat1 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\hmdb\clap.csv')
    Mat2 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\hmdb\run.csv')
    Mat3 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\hmdb\walk.csv')
    Mat4 = R_xcsv(r'C:\Users\86182\Desktop\Spring\KTH-HMDB51\dataset\hmdb\wave.csv')

    totalMat = np.concatenate((Mat1[0:10080], Mat2, Mat3[0:16000], Mat4[0:7140]), axis=0)  # 垂直组合numpy
    # trainingMat=np.concatenate((trainX1,trainX2,trainX3),axis=0)#[10090, 16040, 51857, 7149]# 85136

    Labels1 = [0] * 504#10080  10090
    Labels2 = [1] * 802# 16040
    # Labels3 = [2] * 2592# 51840 51857
    Labels3 = [2] * 800  # 51840 51857
    Labels4 = [3] * 357 # 7140 7149   total_lable=4255

    total_Labels = np.concatenate((Labels1, Labels2, Labels3, Labels4), axis=0)  # 垂直组合numpy

    # #将读取的标签和向量数据存到列表中，并reshape他的维度为 总量 x 1 x 8 x 8
    # totaldata = np.array(totalMat - 18)  # 对数据进行预处理

    totaldata = np.array(totalMat)
    totaldata = totaldata.reshape((len(total_Labels), 1, 60, 75))  #
    totallable = total_Labels

    # 打乱索引
    # 得到的原始数据类型为列表，需要将列表转化为ndarray，在打乱
    totaldata = np.array(totaldata)
    totalLabels = np.array(total_Labels).reshape(len(totallable), )  # 将标签修改成一维

    index = [i for i in range(len(totaldata))]  # test_data为测试数据
    np.random.shuffle(index)  # 打乱索引
    totaldata = totaldata[index]
    totallable = totalLabels[index]

    # 划分训练集和测试集数据70%训练集，30%测试集
    trainX = totaldata[0:1478]#
    train_Labels = totallable[0:1478]

    testX = totaldata[1478:]#2978
    test_Labels = totallable[1478:]

############################################
########得到的保存数据类型要为numpy###########输入为numpy
###########################################
    np.save("target_train_data.npy", trainX)
    np.save('target_train_label.npy',train_Labels)

    np.save("target_test_data.npy", testX)
    np.save('target_test_label.npy',test_Labels)

    trainX = np.load( "target_train_data.npy" )
    ##打印得到的数据就为trainx###