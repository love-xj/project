import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import NearMiss
data = pd.read_excel('data.xls')
#检查是否所有数值都为数字
print(data.applymap(np.isreal).all(axis=0))
data = data.values
y = data[:,-1]
x = data[:,0:-1]
x_feature = () #筛选的特征序号,若为空代表不进行筛选
if_split_train_test =  1 #是否划分训练集和测试集，如果不划分，训练集和测试集都将是整个数据集
sampling = 0 #是否使用采样技术，0代表不使用采样技术，1代表使用欠采样，2代表使用过采样
if len(x_feature) != 0:
	x = x[:,x_feature]
if if_split_train_test:
	x_train,x_test,y_train,y_test = train_test_split(x,y,stratify=y,random_state=42,test_size=0.3)
else:
	x_train = x_test = x
	y_train = y_test = y
if sampling == 1:
	nm = NearMiss(version=1)
	x_train,y_train = nm.fit_sample(x_train,y_train)
elif sampling == 2:
	sm = BorderlineSMOTE(random_state=42,kind="borderline-1")
	x_train,y_train = sm.fit_sample(x_train,y_train)
np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)
np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)
