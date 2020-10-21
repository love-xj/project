import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score 
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool
import geatpy as ea
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold 
x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
penalty = str(np.load('penalty.npy'))
n_splits = int(np.load('n_splits.npy'))
scoring = str(np.load('scoring.npy'))
class MyProblem(ea.Problem): # 继承Problem父类
	def __init__(self, PoolType): # PoolType是取值为'Process'或'Thread'的字符串
		name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
		M = 1 # 初始化M（目标维数）
		maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
		Dim = 1 # 初始化Dim（决策变量维数）
		varTypes = [0] # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
		lb = [2**(-8)] * Dim # 决策变量下界
		ub = [2**8] * Dim # 决策变量上界
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
	i = args[0]
	Vars = args[1]
	data = args[2]
	dataTarget = args[3]
	C = Vars[i, 0]
	# 创建分类器对象并用训练集的数据拟合分类器模型
	model = LogisticRegression(C=C,solver='liblinear',
	penalty=penalty,max_iter=10000).fit(data, dataTarget) 
	if n_splits > 0:
		cv = KFold(n_splits=n_splits,shuffle=False)
	else:
		cv = LeaveOneOut()
	scores = cross_val_score(model,data,dataTarget,cv=cv,scoring=scoring) # 计算交叉验证的得分
	ObjV_i = [scores.mean()] # 把交叉验证的平均得分作为目标函数值
	return ObjV_i  
def run():	
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
	C = var_trace[best_gen,0]
	model = LogisticRegression(C=C,solver='liblinear',penalty=penalty,max_iter=10000)
	model.fit(x_train,y_train)
	return model
def plot(model):
	strs = ['f'+str(i) for i in range(x_train.shape[1])]
	weight = list(zip(strs,model.coef_[0]))
	weight.sort(key = lambda x:abs(x[1]),reverse=True)
	plt.rcParams['font.sans-serif'] = ['SimHei']
	plt.rcParams['axes.unicode_minus'] = False
	plt.bar(*list(zip(*(weight[0:10]))))
