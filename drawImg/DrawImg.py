# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
#微信收藏
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
x =[1,2,83,2,5,66,]
y1 =[13,22,3,43,5,26,]
y2 = [11,2,3,4,25,36,]
area = 2
plt.fill(x, y1, 'b', x, y2, 'r', alpha=0.3)#画普通有填充图像
plt.scatter(x, y1, s=area, alpha=0.5)#聚类图

ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
cset = ax.contourf(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)#画3D图

#画矢量场
plt.streamplot(X, Y, U, V, color=U, linewidth=2, cmap=plt.cm.autumn)
plt.colorbar()




plt.show()


