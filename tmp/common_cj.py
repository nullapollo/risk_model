import pandas as pd
import sklearn.metrics as mtrs
from local.woe import get_woe,get_woe_data
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy as spy
import matplotlib.pyplot as plt


def df_describe(df,excel,workname):
	df1=df0.describe()
	df1.to_excel(excel+".xlsx", sheet_name=workname)	

	
	
	
	
def  cj_woe_iv(df,excel_name,workname):
	from local.woe import get_woe,get_woe_data
	import pandas as pd
	var_cnt=len(df.columns)-2
	model_y=df.columns[1]
	df.rename(columns={model_y:'target'},inplace=True)
	cols=df.columns[(-1*var_cnt):]
	df_frame=pd.DataFrame(columns=['group', 'total', 'bad', 'good', 'bad_pcnt', 'good_pcnt', 'WOE', 'IV', 'sum_iv', 'col'])
	cof=list(df)[(-1*(var_cnt+1)):]
	df4=get_woe_data(df,df_frame,cof)
	df4.to_excel(excel_name+".xlsx", sheet_name=workname)




def  cj_train_test_split(df,test_rate,random_num):
	from sklearn.model_selection import train_test_split    
	model_x=df[df.columns[2:]]
	model_y=df[df.columns[1]]        
	X_train,X_test,y_train,y_test=train_test_split(model_x,model_y,test_size=test_rate, random_state=random_num)       
	return X_train,X_test,y_train,y_test





def  ks_auc_plot(df,test_rate,random_num):
#ks检验函数定义---------------------------------------------------
	import sklearn.metrics as mtrs
	from local.woe import get_woe,get_woe_data
	import matplotlib.pyplot as plt
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report
	import pandas as pd
	import numpy as np
	import matplotlib as mpl
	import scipy as spy
	import matplotlib.pyplot as plt
# compute ks
	def compute_ks(y_true, y_pred):
		fpr, tpr, thresholds = mtrs.roc_curve(y_true, y_pred)
		ks = (tpr - fpr).max()
		return ks

# plot KS curve
	def plot_ks(y_true, y_pred):
		fpr, tpr, thresholds = mtrs.roc_curve(y_true, y_pred)
		ks = (tpr - fpr).max()
		idx = list(tpr - fpr).index(ks)
		n_sample = len(tpr)
		x_axis = [float(i)/n_sample for i in range(n_sample)]
		plt.plot(x_axis, tpr, 'r')
		plt.plot(x_axis, fpr, 'b')
		plt.plot([idx/n_sample, idx/n_sample], [0, 1], 'g--', 
				label=('quantile: '+str(round(idx/n_sample,3))))
		plt.xlabel('Quantile of model score (high -> low)')
		plt.ylabel('Cumulative capture rate')
		plt.title('KS value (' + str(round(ks, 3)) + ')')
		plt.grid(True)
		plt.show()

#============================================================================
	t=3
	KS_value=[]
	AUC_value=[]	
	KS_value1=[]
	AUC_value1=[]	
	df1=df0.fillna(0)
	df1.rename(columns={'y2':'target'},inplace=True)	
	var_cnt=len(df1.columns)-2
	cols=df1.columns[(-1*var_cnt):]
	data2=get_woe(df1,cols)
	dol=data2.columns[3:]
	pred=[]
	for col in dol:
		if col[-3:]=='woe' and data2[col].nunique()>1:
			pred.append(col)
		else :
			continue
	for i in range(t):	    
		X=data2[pred]
		X=X.fillna(0)
    #X=df[['a.gender','a.agebin','a.edu','a.kids','a.income']]
    #X= new1.drop(['MCHT_NO','goal','score_fen','brh_fen','count_suc_amt_fen'], axis=1)
		y=data2['target']
    #n=np.shape(new1)[1]  #特征总维数，包括客户号和分类标签
    #X_lda,y_lda=data_lr.iloc[:,1:n-1].values,data_lr.iloc[:,n-1].values #划分特征与标签
		X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_rate, random_state=random_num)

    #=========================LogisticRegression==============================

		from sklearn.linear_model import LogisticRegression
		classifier = LogisticRegression() #liblinear,lbfgs,newton-cg,sag
		classifier.fit(X_train, y_train)
    #模型评估，计算准去率、召回率等指标
    # predict()函数会自动把阀值设置为0.5
    # 计算在训练集中正确预测的准确率
		y_predicted_train = classifier.predict(X_train)
		#计算在测试集中正确预测的准确率
		y_predicted_test = classifier.predict(X_test)

    ###########################################################################
    #auc值检验
		y_test_prob = classifier.predict_proba(X_test)[:,1]
		from sklearn.metrics import roc_curve, auc 
		fpr, tpr, thresholds = roc_curve(y_test, y_test_prob)
		roc_auc = auc(fpr, tpr)
		ks = compute_ks(y_test, y_test_prob)
		KS_value.append(ks)
		AUC_value.append(roc_auc)

        
        
                   
		y_train_prob = classifier.predict_proba(X_train)[:,1]
		fpr1, tpr1, thresholds1 = roc_curve(y_train, y_train_prob)
		roc_auc1 = auc(fpr1, tpr1)    
		ks1 = compute_ks(y_train, y_train_prob)
		KS_value1.append(ks)
		AUC_value1.append(roc_auc1)
#plt.plot(fpr1, tpr1,'b')
	plot_ks(y_train, y_train_prob)
	plot_ks(y_test, y_test_prob)
#plt.plot(fpr, tpr,'b')
	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr1, tpr1, color='darkorange',
		lw=lw, label='ROC curve (area = %0.2f)' % roc_auc1) ###假正率为横坐标，真正率为纵坐标做曲线
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('training_Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()


	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('test_Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()	
	
	
	
	
	
#########################################################################################变量woe后映射的新数据集与woe化后的新的变量名：data2,pred=model_woe_df(df0)
#df第一列是主键，第二列是y值，后面是入模X值
def  model_woe_df(df):
	df1=df.fillna(0)
	df1.rename(columns={'y2':'target'},inplace=True)	
	var_cnt=len(df1.columns)-2
	cols=df1.columns[(-1*var_cnt):]
	data2=get_woe(df1,cols)
	dol=data2.columns[3:]
	pred=[]
	for col in dol:
		if col[-3:]=='woe' and data2[col].nunique()>1:
			pred.append(col)
		else :
			continue
	return data2,pred
	
#########################################################################################计算KS值：ks=compute_ks(y_true, y_pred)
def compute_ks(y_true, y_pred):
	fpr, tpr, thresholds = mtrs.roc_curve(y_true, y_pred)
	ks = (tpr - fpr).max()
	return ks


	
##########################################################################################绘制KS曲线图:plot_ks(y_true, y_pred)
def plot_ks(y_true, y_pred):
	fpr, tpr, thresholds = mtrs.roc_curve(y_true, y_pred)
	ks = (tpr - fpr).max()
	idx = list(tpr - fpr).index(ks)
	n_sample = len(tpr)
	x_axis = [float(i)/n_sample for i in range(n_sample)]
	plt.plot(x_axis, tpr, 'r')
	plt.plot(x_axis, fpr, 'b')
	plt.plot([idx/n_sample, idx/n_sample], [0, 1], 'g--', 
			label=('quantile: '+str(round(idx/n_sample,3))))
	plt.xlabel('Quantile of model score (high -> low)')
	plt.ylabel('Cumulative capture rate')
	plt.title('KS value (' + str(round(ks, 3)) + ')')
	plt.grid(True)
	plt.show()

	
##########################################################################################绘制auc曲线图:plot_auc(fpr,tpr,roc_auc)
def plot_auc(fpr,tpr,roc_auc):
	plt.figure()
	lw = 2
	plt.figure(figsize=(10,10))
	plt.plot(fpr, tpr, color='darkorange',
		lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	plt.show()	
	
	
	



def plot_decisiontree(df,dot_name,splitter_method="best",n_max_depth=None,n_min_samples_split=2,n_min_samples_leaf=1,n_max_features=None,n_random_state=None,n_max_leaf_nodes=None,n_class_weight=None,):
	#画出决策树
	from sklearn import tree
	from sklearn.datasets import load_wine
	from sklearn.model_selection import train_test_split
	import pandas as pd
	import numpy as np
	from sklearn.tree import export_graphviz
	model_x=df[df.columns[2:]]
	model_y=df[df.columns[1]]   
	xtrain,xtest,ytrain,ytest=train_test_split(model_x,model_y,test_size=0.3)
	# 构建模型
	clf=tree.DecisionTreeClassifier(splitter=splitter_method,
		max_depth=n_max_depth,
		min_samples_split=n_min_samples_split,
		min_samples_leaf=n_min_samples_leaf,
		max_features=n_max_features,
		random_state=n_random_state,
		max_leaf_nodes=n_max_leaf_nodes,
		class_weight=n_class_weight)
	clf.fit(xtrain,ytrain)
	with open(dot_name+'.dot','w',encoding='utf-8') as f:
		f=export_graphviz(clf,feature_names=df.columns[2:],out_file=f)








	
	
	
	
	
	
	
	
	