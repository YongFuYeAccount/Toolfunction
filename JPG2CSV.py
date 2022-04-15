# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import csv,os,cv2
from os import listdir

#定义读取文件夹里文件名的文件函数
def getlist(Filename):
    FileList = listdir(Filename)
    return FileList

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


def convert_img_to_csv(img_dir):
    #设置需要保存的csv路径
    with open(r"C:\Users\86182\Desktop\Spring\Public2private\test.csv",'w',newline='') as f:
        ###########设置csv文件的列名#########
        # column_name = ["label"]
        # column_name.extend(["pixel%d"%i for i in range(32*32)])
        #####将列名写入到csv文件中########
        # writer = csv.writer(f)
        # writer.writerow(column_name)
        #该目录下有9个目录,目录名从0-9
        c = getlist(r'C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\Init_Digits\testing_image')
        for i in range(len(c)):
            #获取目录的路径
            img_temp_dir = os.path.join(img_dir,c[i])
            #获取该目录下所有的文件
            img_list = os.listdir(img_temp_dir)
            #遍历所有的文件名称
            for img_name in img_list:
                #判断文件是否为目录,如果为目录则不处理
                if not os.path.isdir(img_name):
                    #获取图片的路径
                    img_path = os.path.join(img_temp_dir,img_name)
                    #因为图片是黑白的，所以以灰色读取图片
                    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)

                    # res = cv2.resize(img, (100, 100))
                    #图片标签
                    row_data = [i]

                    # Normalize(res)d#对数据进行处理
                    #获取图片的像素
                    row_data.extend(img.flatten())
                    #将图片数据写入到csv文件中
                    writer = csv.writer(f)
                    writer.writerow(row_data)


if __name__ == "__main__":
    #将该目录下的图片保存为csv文件
    convert_img_to_csv(r"C:\Users\86182\Desktop\GraduateDesign\Aug\Infrared image\Init_Digits\testing_image")
