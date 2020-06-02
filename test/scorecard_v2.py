#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 14:19
# @Author  : Andrewma
# @Site    : 
# @File    : scorecard_v2.py
# @Software: PyCharm


from sklearn.model_selection import train_test_split

from crsc.utils.mtools import *
from .preprocess_function import *


"""
***************************************************************************************************
Step 0: 数据集提取
1. 读入数据
2. 选择合适的建模样本
3. 数据集划分成训练集和测试集
***************************************************************************************************
"""

allData = pd.read_csv('./Data/application.csv', header=0, encoding='latin1')
allData['term'] = allData['term'].map(lambda x: int(x.replace(' months', '')))

# 处理标签：Fully Paid是正常用户；Charged Off是违约用户
allData['y'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))

# 由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取'term＝36months'的行本
allData1 = allData.loc[allData.term == 36]

trainData, testData = train_test_split(allData1, test_size=0.4)


# TODO: 差个数据摸底的步骤，数据的缺失情况、分布情况，后面也是必要的


"""
***************************************************************************************************
Step 1: 数据预处理，包括：
    1. 数据清洗
    2. 格式转换
    3. 缺失值填补
***************************************************************************************************
"""

# 将带％的百分比变为浮点数(string 2 float)
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%', ''))/100.)

# 将工作年限进行转化，否则影响排序（新增数据列，不影响原始数据集数据）
trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)

# 将desc的缺失作为一种状态，非缺失作为另一种状态
trainData['desc_clean'] = trainData['desc'].map(DescExisting)

# 处理日期 earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x, '%b-%y'))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x, '%b-%y'))

# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x: MakeupMissing(x))

trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x: MakeupMissing(x))

trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x: MakeupMissing(x))


"""
***************************************************************************************************
Step 2: 变量衍生(专家经验方法，非GBDT衍生变量)
***************************************************************************************************
"""

# 考虑申请额度与收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis=1)

# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(
    lambda x: MonthGap(x.earliest_cr_line_clean, x.app_date_clean), axis=1)


"""
***************************************************************************************************
Step 3: 数据分箱
采用ChiMerge, 要求分箱完之后：
（1）不超过5箱
（2）Bad Rate单调
（3）每箱同时包含好坏样本
（4）特殊值如－1，单独成一箱

连续型变量：可直接分箱
类别型变量：
（a）当取值较多时，先用bad rate编码，再用连续型分箱的方式进行分箱
（b）当取值较少时：
    （b1）如果每种类别同时包含好坏样本，无需分箱
    （b2）如果有类别只包含好坏样本的一种，需要合并
***************************************************************************************************
"""


# 数值型变量
num_features = [
    'int_rate_clean',
    'emp_length_clean',
    'annual_inc',
    'dti',
    'delinq_2yrs',
    'earliest_cr_to_app',
    'inq_last_6mths',
    'mths_since_last_record_clean',
    'mths_since_last_delinq_clean',
    'open_acc',
    'pub_rec',
    'total_acc'
]

# 类别型变量
cat_features = [
    'home_ownership',
    'verification_status',
    'desc_clean',
    'purpose',
    'zip_code',
    'addr_state',
    'pub_rec_bankruptcies_clean'
]


(df2, cat_info) = CategoryBinSplit(trainData, 'y', cat_features, max_bins=5)

num_features += cat_info['br_encoding_features']

(df3, con_info) = ContinuousBinSplit(df2, 'y', num_features, max_bins=5, special_attribute=[-1])

all_var = cat_info['less_value_features'] + \
          cat_info['less_value_merge_features'] + \
          con_info['continuous_merge_features']


"""
***************************************************************************************************
Step 4：WOE编码、计算IV
***************************************************************************************************
"""

(WOE, IV) = FeaturesForecastImportance(df3, 'y', all_var, plot_flag=True)

IV_selected = IV.loc[IV.iv >= 0.01]

WOE_trans_features = []
for var in list(IV_selected.index):
    new_var = var + '_WOE'
    df3[new_var] = df3[var].map(WOE[var])
    WOE_trans_features.append(new_var)


"""
***************************************************************************************************
Step 5：相关性检验，多重共线性分析
***************************************************************************************************
"""
(clear_features, delete_features) = FeaturesCorrelationAnalysis(df3, list(IV_selected.index), IV)
(VIF, VIF_list) = FeaturesVIFAnalysis(df3, clear_features)


"""
***************************************************************************************************
Step 6： 建立Logistics回归模型
***************************************************************************************************
"""


