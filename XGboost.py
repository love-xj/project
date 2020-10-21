import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score 
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import geatpy as ea
from xgboost import plot_importance
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
n_splits = int(np.load('n_splits.npy'))
scoring = str(np.load('scoring.npy'))
max_depth = 0
min_child_weight = 0
class MyProblem(ea.Problem): # 继承Problem父类
	def __init__(self, PoolType): # PoolType是取值为'Process'或'Thread'的字符串
		name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
		M = 1 # 初始化M（目标维数）
		maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
		Dim = 3 # 初始化Dim（决策变量维数）
		varTypes = [0,0,0] # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
		lb = [0,0.6,0.6]  # 决策变量下界
		ub = [0.5,1,1]  # 决策变量上界
		lbin = [1] * Dim # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
		ubin = [1] * Dim # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
		# 调用父类构造方法完成实例化
		ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
		self.data = x_train
		self.dataTarget = y_train
		# 设置用多线程还是多进程
		self.PoolType = PoolType
		if self.PoolType == 'Thread':
			self.pool = ThreadPool(2) # 设置池的大小
		elif self.PoolType == 'Process':
			num_cores = int(mp.cpu_count()) # 获得计算机的核心数
			self.pool = ProcessPool(num_cores) # 设置池的大小
	def aimFunc(self, pop): # 目标函数，采用多线程加速计算
		Vars = pop.Phen # 得到决策变量矩阵
		args = list(zip(list(range(pop.sizes)), [Vars] * pop.sizes, [self.data] * pop.sizes, [self.dataTarget] * pop.sizes))
		if self.PoolType == 'Thread':
			pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
		elif self.PoolType == 'Process':
			result = self.pool.map_async(subAimFunc, args)
			result.wait()
			pop.ObjV = np.array(result.get())
def subAimFunc(args):
	max_depth = int(np.load('max_depth.npy'))
	min_child_weight = int(np.load('min_child_weight.npy'))
	i = args[0]
	Vars = args[1]
	data = args[2]
	dataTarget = args[3]
	gamma = Vars[i,0]
	subsample = Vars[i,1]
	colsample_bytree = Vars[i,2]
	model = XGBClassifier(silent=1,max_depth=max_depth,min_child_weight=min_child_weight,gamma=gamma,
		subsample=subsample,colsample_bytree=colsample_bytree)
	model.fit(x_train,y_train)
	if n_splits > 0:
		cv = KFold(n_splits=n_splits,shuffle=False,random_state=0)
	else:
		cv = LeaveOneOut()	
	scores = cross_val_score(model,data,dataTarget, cv=cv,scoring=scoring) # 计算交叉验证的得分
	ObjV_i = [scores.mean()] # 把交叉验证的平均得分作为目标函数值
	return ObjV_i	  
def run():
	#第一步网格搜索
	if n_splits > 0:
		cv = StratifiedKFold(n_splits=n_splits,shuffle=False,random_state=0)
	else:
		cv = LeaveOneOut()
	param = {'max_depth':range(3,11),'min_child_weight':range(1,7)}
	model = XGBClassifier(silent=1)
	grid_search = GridSearchCV(model,param,cv=cv,scoring=scoring)
	grid_search.fit(x_train,y_train)
	print(str(grid_search.best_params_) + ' ' + str(grid_search.best_score_))
	max_depth = grid_search.best_params_['max_depth']
	min_child_weight = grid_search.best_params_['min_child_weight']	
	np.save('max_depth.npy',max_depth)
	np.save('min_child_weight.npy',min_child_weight)
	#第二步用遗传算法进行搜索
	"""===============================实例化问题对象==========================="""
	PoolType = 'Process' # 设置采用多进程，若修改为: PoolType = 'Thread'，则表示用多线程
	problem = MyProblem(PoolType) # 生成问题对象
	"""=================================种群设置==============================="""
	Encoding = 'RI'	   # 编码方式
	NIND = 50			 # 种群规模
	Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
	population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
	"""===============================算法参数设置============================="""
	myAlgorithm = ea.soea_DE_rand_1_bin_templet(problem, population) # 实例化一个算法模板对象
	myAlgorithm.MAXGEN = 30 # 最大进化代数
	myAlgorithm.trappedValue = 1e-6 # “进化停滞”判断阈值
	myAlgorithm.maxTrappedCount = 10 # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
	"""==========================调用算法模板进行种群进化======================="""
	[population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
	population.save() # 把最后一代种群的信息保存到文件中
	problem.pool.close() # 及时关闭问题类中的池，否则在采用多进程运算后内存得不到释放
	# 输出结果
	best_gen = np.argmin(problem.maxormins * obj_trace[:, 1]) # 记录最优种群个体是在哪一代
	best_ObjV = obj_trace[best_gen, 1]
	print('最优的目标函数值为：%s'%(best_ObjV))
	print('最优的控制变量值为：')
	for i in range(var_trace.shape[1]):
		print(var_trace[best_gen, i])
	print('有效进化代数：%s'%(obj_trace.shape[0]))
	print('最优的一代是第 %s 代'%(best_gen + 1))
	print('评价次数：%s'%(myAlgorithm.evalsNum))
	print('时间已过 %s 秒'%(myAlgorithm.passTime))
	gamma = var_trace[best_gen,0]
	subsample = var_trace[best_gen,1]
	colsample_bytree = var_trace[best_gen,2]
	model = XGBClassifier(silent=1,max_depth=max_depth,min_child_weight=min_child_weight,gamma=gamma,
		subsample=subsample,colsample_bytree=colsample_bytree)
	model.fit(x_train,y_train)
	return model
def plot(model):
	plot_importance(model)