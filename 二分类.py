import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
#二分类机器学习类
class ML():
	def __init__(self,algorithm='randomforest',n_splits=10,scoring='accuracy',penalty='l1'):
		'''
		x是自变量，y是因变量
		algorithm为使用的机器学习算法，默认随机森林
		目前支持几种：
			'randomforest':随机森林
			'logistic':logistic回归
				如果指定为logistic回归，还可通过penalty参数指定正则化方式，默认为l1正则化
			'xgboost':xgboost
			'SVM':支持向量机
		n_splits是交叉验证的折数，默认为10，如果设置为0，代表使用留一法
		scoring是模型选择的评估指标，默认为准确度
		目前支持以下几种：
			'accuracy':准确性
			'roc_auc':ROC 曲线下方的面积,也被称为AUC
			'average_precision':准确率-召回率曲线下方的面积
		'''
		self.algorithm = algorithm
		np.save('n_splits.npy',n_splits)
		np.save('scoring.npy',scoring)
		np.save('penalty.npy',penalty)
		self.x_test = np.load('x_test.npy')
		self.y_test = np.load('y_test.npy')
	def run(self):
		if self.algorithm == 'logistic':
			import logistic
			self.model = logistic.run()
			self.name = 'LogisticRegression'
		elif self.algorithm == 'randomforest':
			import randomforest
			self.model = randomforest.run()
			self.name = 'RandomForest'
		elif self.algorithm == 'xgboost':
			import XGboost
			self.model = XGboost.run()
			self.name = 'xgboost'
		elif self.algorithm == 'SVM':
			import SVM
			self.model = SVM.run()
			self.x_test_scaled = np.load('x_test_scaled.npy')
			self.name = 'SVM'
		else:
			print('算法参数有误')
		if self.algorithm == 'SVM':
			self.y_pred_prob = self.model.decision_function(self.x_test_scaled)
			self.y_pred = self.model.predict(self.x_test_scaled)
		else:
			self.y_pred_prob = self.model.predict_proba(self.x_test)[:,1]
			self.y_pred = self.model.predict(self.x_test)
	def result_analysis(self):
		#对结果进行分析
		#混淆矩阵
		self.confusion_matrix = confusion_matrix(self.y_test,self.y_pred)
		self.TP = TP = self.confusion_matrix[1,1] #真阳性
		self.FN = FN = self.confusion_matrix[1,0] #假阴性
		self.FP = FP = self.confusion_matrix[0,1] #假阳性
		self.TN = TN = self.confusion_matrix[0,0] #真阴性
		self.accuracy = (TP + TN) / (TP + TN + FP + FN)
		print("模型的准确率(accuracy)为:",end='')
		print(self.accuracy)
		self.precision = TP / (TP + FP) #精确率
		print("模型的精确率(precision)为:",end='')
		print(self.precision)
		self.recall = TP / (TP + FN) #召回率
		print("模型的召回率(recall)为:",end='')
		print(self.recall)
		self.PPV = self.precision #阳性预测值
		print("模型的阳性预测值(PPV)为:",end='')
		print(self.PPV)
		self.NPV = TN / (TN + FN) #阴性预测值
		print("模型的阴性预测值(NPV)为:",end='')
		print(self.NPV)
		self.sensitivity = self.recall #灵敏度
		print("模型的灵敏度(sensitivity)为:",end='')
		print(self.sensitivity)
		self.specificity = TN / (TN + FP) #特异度
		print("模型的特异度(specificity)为:",end='')
		print(self.specificity)
	def plot_importance(self):
		#画出重要性排序图
		if self.algorithm == 'logistic':
			import logistic
			logistic.plot(self.model)
		elif self.algorithm == 'randomforest':
			import randomforest
			randomforest.plot(self.model)
		elif self.algorithm == 'xgboost':
			import XGboost
			XGboost.plot(self.model)
		elif self.algorithm == 'SVM':
			print('支持向量机算法不能画出重要性排序图')
	def plot_roc(self):
		#画出roc曲线
		fpr,tpr,thresholds = roc_curve(self.y_test,self.y_pred_prob)
		self.roc_auc = auc(fpr,tpr)
		label = self.name+ '(auc=' + str(round(self.roc_auc,3)) + ')'
		plt.plot(fpr,tpr,label=label)
		plt.xlabel('False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.plot([0,1],[0,1],'r--',color='navy')
		plt.title('Receiver operating characteristic example')
		plt.legend(loc='best')
		print("模型的AUROC为:",end='')
		print(self.roc_auc)
	def plot_PR(self):
		#画出PR曲线
		precision,recall,thresholds = precision_recall_curve(self.y_test,self.y_pred_prob)
		self.PR_auc = auc(recall,precision)
		label = self.name+ '(auc=' + str(round(self.PR_auc,3)) + ')'
		plt.plot(recall,precision,label=label)
		plt.xlabel('Recall Rate')
		plt.ylabel('Precision Rate')
		plt.plot([0,1],[0,1],'r--',color='navy')
		plt.title('Precision-recall curve')
		plt.legend(loc='best')
		print("模型的AUPRC为:",end='')
		print(self.PR_auc)
if __name__ == '__main__':
	n_splits = 5
	scoring = 'precision'
	model1 = ML('logistic',n_splits=n_splits,scoring=scoring)
	model1.run()
	print("以下为logistic回归模型结果:")
	model1.result_analysis()
	model1.plot_importance()
	plt.savefig('logistic_重要性')
	plt.clf()
	model2 = ML(n_splits=n_splits,scoring=scoring)
	model2.run()
	print("以下为随机森林模型结果:")
	model2.result_analysis()
	model2.plot_importance()
	plt.savefig('随进森林_重要性')
	plt.clf()
	model3 = ML('SVM',n_splits=n_splits,scoring=scoring)
	model3.run()
	print("以下为SVM模型结果:")
	model3.result_analysis()
	model3.plot_importance()
	model4 = ML('xgboost',n_splits=n_splits,scoring=scoring)
	model4.run()
	print("以下为xgboost模型结果:")
	model4.result_analysis()
	model4.plot_importance()
	plt.savefig('xgboost_重要性')
	plt.clf()
	model1.plot_roc()
	model2.plot_roc()
	model3.plot_roc()
	model4.plot_roc()
	plt.savefig('AUROC')
	plt.clf()
	model1.plot_PR()
	model2.plot_PR()
	model3.plot_PR()
	model4.plot_PR()
	plt.savefig('AUPRC')
	plt.clf()


