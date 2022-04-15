

import pandas as pd
import os

# data = pd.read_csv(r"C:\Users\86182\Desktop\ToolsPY\test_submit.csv"), #
# with open(r"C:\Users\86182\Desktop\ToolsPY\news_data.txt", 'a+') as f:#, encoding='utf-8'
#     for line in data.values:
#         f.write((str(line[0]) + '\t' + str(line[1]) + '\n'))
#


#
data = pd.read_csv(r"C:\Users\86182\Desktop\ToolsPY\test0.csv",encoding='gbk')#有中文的时候的编码用gbk，encoding='utf-8'
with open(r"C:\Users\86182\Desktop\ToolsPY\news_data.txt", 'w',encoding='gbk') as f:
    for line in data.values:
        f.write((str(line[0]) + '\t' + str(line[1]) + '\t' + str(line[2]) + '\t' + str(line[3]) + '\t' + str(line[4])))#写csv中文件的五列
        f.write("\n")