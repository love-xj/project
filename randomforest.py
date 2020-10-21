from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold 
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
n_splits = int(np.load('n_splits.npy'))
scoring = str(np.load('scoring.npy'))
def run():
	if n_splits > 0:
		cv = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=0)
	else:
		cv = LeaveOneOut()
	param = {'n_estimators':range(1,211,10)}
	model = RandomForestClassifier(random_state=0)
	grid_search = GridSearchCV(model,param,cv=cv,scoring=scoring)
	grid_search.fit(x_train,y_train)
	print(str(grid_search.best_params_) + ' ' + str(grid_search.best_score_))
	#第二轮搜索
	n_estimators = grid_search.best_params_['n_estimators']
	min_n = max(1,n_estimators-9)
	max_n = min(201,n_estimators+9)
	param = {'n_estimators':range(min_n,max_n)}
	grid_search = GridSearchCV(model,param,cv=cv,scoring=scoring)
	grid_search.fit(x_train,y_train)
	print(str(grid_search.best_params_) + ' ' + str(grid_search.best_score_))
	#最终结果
	n_estimators = grid_search.best_params_['n_estimators']
	model = RandomForestClassifier(n_estimators=n_estimators,random_state=0)
	model.fit(x_train,y_train)
	return model
def plot(model):
	strs = ['f'+str(i) for i in range(x_train.shape[1])]
	weight = list(zip(strs,model.feature_importances_))
	weight.sort(key = lambda x:abs(x[1]),reverse=True)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	plt.bar(*list(zip(*(weight[0:20]))))