

import os
a = [1,2,5,4,8]
b = [1,2,5,4,8]
c = []
# dataframe = pd.DataFrame({'epoch':a,'acc_epoch':b})
# dataframe.to_csv(r"C:\Users\86182\Desktop\LearnText\pytorch\test\practice\LSTM\Preprocess\stand\adjust parameter\stand_epoch_acc_train_val_loss.csv",index=False,sep=',')

filepath = r"C:\Users\86182\Desktop\freight station"
for i in range(2):
    print(str(i))

    path = os.path.join(filepath,str(i))
    path1 = path +  '.csv'
    dataframe = pd.DataFrame({'epoch':a,'acc_epoch':b})
    dataframe.to_csv(path1)
# print("1:",os.path.join('aaaa','/bbbb','ccccc.txt'))
# print("2:",os.path.join('/aaaa','/bbbb','/ccccc.txt'))  #不良写法习惯
# print("3:",os.path.join('aaaa','./bbb','ccccc.txt'))