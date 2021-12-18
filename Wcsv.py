# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 09:29:57 2021

@author: 86182
"""

"""
列表按列写入
"""
import pandas as pd

a = [1,2,3]
b= [3,4,4]
c =[5,5,5]
#字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name':a,'b_name':b,'c_name':c})
dataframe.to_csv('test.csv',index=False,sep=',')




import csv
dic = {'WWWW': 'FFFF', 'MSS': 2, 'TTL': 40, 'WS': 3, 'S': 1, 'N': 1, 'D': 0, 'T': 8, 'F': 'S', 'LEN': '3C'}
print(dic.items())
with open('test.csv','w',newline='') as f:
    writer = csv.writer(f)
    for row in dic.items():
        writer.writerow(row)



#将DataFrame存储为csv,index表示是否显示行名，default=True 分隔符选用分号
dataframe.to_csv(r"PaperList.csv",index=False,sep=';',encoding= 'utf-8-sig')
                                #encoding = 'utf-8-sig使得excel不会乱码


                            
    
    
    
    
    
    
    
