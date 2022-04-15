# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np


def drawPre_recall(test_y, y_score):#test_y为测试数据的类别，y_score是每个类别的概率
    y = test_y
    # 使用label_binarize让数据成为类似多标签的设置
    Y = label_binarize(y, classes=[0, 1, 2, 3, 4])
    n_classes = Y.shape[1]

    # 对每个类别
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y[:, i],y_score[:, i])
        average_precision[i] = average_precision_score(Y[:, i], y_score[:, i])

    # 一个"微观平均": 共同量化所有课程的分数
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y.ravel(),y_score.ravel())

    average_precision["micro"] = average_precision_score(Y, y_score, average="micro")

    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.01])
    plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.show()
