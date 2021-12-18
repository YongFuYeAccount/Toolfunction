# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import csv
import numpy as np
import matplotlib.pyplot as plt

score_path = r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Reload\ClassificateOdds.txt" # csv文件路径
with open(score_path, 'r') as f:
    files = f.readlines()      # 读取文件

lis_all = []
for file in files:
    # print(file.strip().split(" "))
    _, _, s1, s2, s3, s4,s5 = file.strip().split("\t")
    lis_all.append(s1)
    lis_all.append(s2)
    lis_all.append(s3)
    lis_all.append(s4)
    lis_all.append(s5)
lis_order = sorted(set(lis_all))   # 记录所有得分情况，并去重从小到大排序，寻找各个阈值点

micro_precis = []
micro_recall = []

for i in lis_order:

    true_p0 = 0         # 真阳
    true_n0 = 0         # 真阴
    false_p0 = 0        # 假阳
    false_n0 = 0        # 假阴

    true_p1 = 0
    true_n1 = 0
    false_p1 = 0
    false_n1 = 0

    true_p2 = 0
    true_n2 = 0
    false_p2 = 0
    false_n2 = 0

    true_p3 = 0
    true_n3 = 0
    false_p3 = 0
    false_n3 = 0

    true_p4 = 0
    true_n4 = 0
    false_p4 = 0
    false_n4 = 0
    for file in files:
        cls, pd, n0, n1, n2, n3, n4 = file.strip().split("\t")       # 分别计算比较各个类别的得分，分开计算，各自为二分类，
                                                            # 最后求平均，得出宏pr

        if float(n0) >= float(i) and cls == '0':               # 遍历所有样本，第0类为正样本，其他类为负样本，
            true_p0 = true_p0 + 1                              # 大于等于阈值，并且真实为正样本，即为真阳，
        elif float(n0) >= float(i) and cls != '0':             # 大于等于阈值，真实为负样本，即为假阳；
            false_p0 = false_p0 + 1                            # 小于阈值，真实为正样本，即为假阴
        elif float(n0) < float(i) and cls == '0':
            false_n0 = false_n0 + 1

        if float(n1) >= float(i) and cls == '1':                # 遍历所有样本，第1类为正样本，其他类为负样本
            true_p1 = true_p1 + 1
        elif float(n1) >= float(i) and cls != '1':
            false_p1 = false_p1 + 1
        elif float(n1) < float(i) and cls == '1':
            false_n1 = false_n1 + 1

        if float(n2) >= float(i) and cls == '2':                # 遍历所有样本，第2类为正样本，其他类为负样本
            true_p2 = true_p2 + 1
        elif float(n2) >= float(i) and cls != '2':
            false_p2 = false_p2 + 1
        elif float(n2) < float(i) and cls == '2':
            false_n2 = false_n2 + 1


        if float(n3) >= float(i) and cls == '3':                # 遍历所有样本，第3类为正样本，其他类为负样本
            true_p3 = true_p3 + 1
        elif float(n3) >= float(i) and cls != '3':
            false_p3 = false_p3 + 1
        elif float(n2) < float(i) and cls == '3':
            false_n3 = false_n3 + 1


        if float(n4) >= float(i) and cls == '4':                # 遍历所有样本，第4类为正样本，其他类为负样本
            true_p4 = true_p4 + 1
        elif float(n4) >= float(i) and cls != '4':
            false_p4 = false_p4 + 1
        elif float(n4) < float(i) and cls == '4':
            false_n4 = false_n4 + 1

    prec0 = (true_p0+0.00000000001) / (true_p0 + false_p0 + 0.000000000001)      # 计算各类别的精确率，小数防止分母为0
    prec1 = (true_p1+0.00000000001) / (true_p1 + false_p1 + 0.000000000001)
    prec2 = (true_p2+0.00000000001) / (true_p2 + false_p2 + 0.000000000001)
    prec3 = (true_p3 + 0.00000000001) / (true_p3 + false_p3 + 0.000000000001)
    prec4 = (true_p4 + 0.00000000001) / (true_p4 + false_p4 + 0.000000000001)



    recall0 = (true_p0+0.00000000001)/(true_p0+false_n0 + 0.000000000001)        # 计算各类别的召回率，小数防止分母为0
    recall1 = (true_p1+0.00000000001) / (true_p1 + false_n1+0.000000000001)
    recall2 = (true_p2+0.00000000001) / (true_p2+false_n2 + 0.00000000001)
    recall3 = (true_p3 + 0.00000000001) / (true_p3 + false_n3 + 0.00000000001)
    recall4= (true_p4 + 0.00000000001) / (true_p4 + false_n4 + 0.00000000001)

    precision = (prec0 + prec1 + prec2 + prec3 + prec4)/5
    recall = (recall0 + recall1 + recall2 + recall3 + recall4)/5              # 多分类求得平均精确度和平均召回率，即宏micro_pr
    micro_precis.append(precision)
    micro_recall.append(recall)

micro_precis.append(1)
micro_recall.append(0)
print(micro_precis)
print(micro_recall)

x = np.array(micro_recall)
y = np.array(micro_precis)
plt.figure()
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('recall')
plt.ylabel('precision')
plt.title('PR curve')
plt.plot(x, y)
plt.show()




