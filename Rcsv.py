# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 16:33:00 2021

@author: 86182
"""

import csv  
def Rcsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数（这里是学生人数）
    i = 0
    while i < renshu:
       print (date[i][3])    #访问列表date中的数据验证读取成功（这里是第i个第三）

    return date
a=Rcsv(r'C:\Users\86182\Desktop\PythonMission\Part2\SummaryIformation.csv')
print(a)

#按照行读取数据
with open(filename,encoding="utf-8") as f:
    reader = csv.reader(f)
    header_row = next(reader)
    datas = []
    for row in reader:
        print(row[2])



#读标签函数
def R_ycsv(Filedirectory):
    csv_file = open(Filedirectory, encoding='utf-8')  # 打开csv文件
    csv_reader_lines = csv.reader(csv_file)  # 逐行读取csv文件
    date = []  # 创建列表准备接收csv各行数据
    renshu = 0
    for one_line in csv_reader_lines:
        date.append(one_line)  # 将读取的csv分行数据按行存入列表‘date’中
        renshu = renshu + 1  # 统计行数

    for i in range(len(date)):#将读取的字符串转化为数值
        date[i] = list(map(int, date[i]))
    # date = np.array(date, dtype=float)  # trainX为待转化的列表
    return date