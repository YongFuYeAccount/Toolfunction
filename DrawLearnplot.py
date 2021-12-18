# -*- coding: utf-8 -*-
#file location the same as spider
# author yyf
import matplotlib.pyplot as plt

loss = [0.9,0.5,0.45,0.4,0.35,0.33,0.3,0.25,0.2,0.1,0.1,0.1,0.1,0.1]
val_loss = [0.5,0.45,0.35,0.3,0.25,0.22,0.24,0.21,0.19,0.2,0.2,0.2,0.2,0.2]


# 绘制训练 & 验证的损失值
plt.plot(loss)
plt.plot(val_loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

