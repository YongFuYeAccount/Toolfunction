import  csv
import torch
import torch.nn as nn
import csv
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# stag_01_submit
csvFile = open(r"C:\Users\86182\Desktop\ToolsPY\test_submit.csv", "w")  # 创建csv文件
writer = csv.writer(csvFile)  # 创建写的对象
# 先写入columns_name
writer.writerow(["实际", "预测", "0", "1", "2", "3", "4"])  # 写入列的名称

#定义模型保存的绝对路径
modelfilepath = r'C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Reload\LSTM.test4.pkl'
net = torch.load(modelfilepath)#加载模型
#测试所有数据，得到输出
test_output = net(input)#与定义的模型输入保持一致
pred_y = torch.max(test_output, 1)[1].data.numpy()#直接得到预测的结果预测的标签(输出是numpy)

#计算softmax概率
probability = torch.nn.functional.softmax(test_output,dim=1)#计算softmax，即该输入数据属于各类的概率
max_value,index = torch.max(probability,1)#找到最大概率对应的索引号，该图片即为该索引号对应的类别（输出是tensor）
pred_class = list(pred_y)#将预测的结果放入列表中

probability = np.round(probability.detach().numpy(), 3)#保留概率的三位小数
totalprobality = probability #得到的probality是np形式
# #准确率
accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
print(accuracy)

test_Labels = list(test_Labels)#将实际标签放入列表中去
for i in range(len(list(test_Labels))):
    probability = list(totalprobality[i])
    writer.writerow(
        [test_Labels[i], pred_class[i], probability[0], probability[1], probability[2], probability[3],
         probability[4]])
csvFile.close()


def keepresult(savefilename, headerlines, real_Labels, pred_class, totalprobality):
        #保存csv的地址，csv的列名，实际的标签（list），预测的标签（list），各个列别的标签(list)
    csvFile = open(savefilename, "w")  # 创建csv文件
    writer = csv.writer(csvFile)  # 创建写的对象
    # 先写入columns_name
    writer.writerow(headerlines)  # 写入列的名称
    for i in range(len(list(real_Labels))):
        probability = list(totalprobality[i])#将概率np转化为list
        writer.writerow(
            [real_Labels[i], pred_class[i], probability[0], probability[1], probability[2], probability[3],
             probability[4]])
    csvFile.close()


