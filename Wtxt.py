# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 16:10:39 2021

@author: 86182
"""

import time
import math

#取当前10位时间戳，返回时间戳和时间显示为列表
def GetN():
    TIME =[]
    time_stamp = int(time.time())
    TIME.append( time_stamp)
    #print(time_stamp)
    Datatime=time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_stamp))
    #print(Datatime)
    TIME.append(Datatime)
    return  TIME

def WdataSTR(data):
    f=open("k1.txt","a")#打开文本，追加数据
    f.writelines('\n')#每次追加数据时换行
    a= GetN()
    f.writelines(str(data))#把data以字符串形式写入
    f.write(',')#列表中的数据全部以‘，’分割开
    for i in range(len(a)):
        a[i] = str(a[i])#将列表中的数据全部转化为字符串
        f.write(a[i])
        f.write(',')#列表中的数据全部以‘，’分割开
    
    f.close()#
    

    
    









    