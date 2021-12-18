

from sklearn.preprocessing import MinMaxScaler
def mm(Range, data):#归一化的范围（tuple），data 是ndarray
    """
    归一化处理
    """
    data = list(data)
    mm = MinMaxScaler(feature_range=Range)
    data = mm.fit_transform(data)
    return data     #返回归一化后的数据
    
    
#定义标准化缩放函数
from sklearn.preprocessing import StandardScaler
def stand(data):#输入data为list
    """
    标准化缩放
    """
    std = StandardScaler()
    data = std.fit_transform(data)
    data = np.round(data, 4)
    return data


#对数据正则化处理
from sklearn.preprocessing import Normalizer
def Normal(data):#输入数据类型为列表
    transformer = Normalizer().fit(data)  # fit does nothing.
    Normalizer(copy=True, norm='l2')#选择l2范数
    data = transformer.transform(data)
    return data
    