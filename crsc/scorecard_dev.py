#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 21:34
# @Author  : Andrewma
# @Site    : 
# @File    : scorelast.py
# @Software: PyCharm

from crsc.scorecard_functions import *
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier


def CategoryBinSplit(df, target, cat_features, max_bins=5):
    """
    # 针对分类变量进行特征分箱处理

    :param df: 包含变量的数据集
    :param target: 分类标签 y
    :param cat_features: 分类变量列表
    :return: 新增分箱特征的数据集df2, 分类变量信息字典
    """

    df2 = df.copy()
    more_value_features = []
    less_value_features = []

    # Step 1：先处理分类变量，检查类别型变量中，哪些变量取值超过5
    for var in cat_features:
        valueCounts = len(set(df2[var]))
        print(var, valueCounts)
        if valueCounts > max_bins:
            more_value_features.append(var)   #取值超过 max_bins 的变量，需要bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)

    # Step 2: 处理水平少于 max_bins 的变量，只用检查是否同时存在正负样本
    merge_bin_dict = {}
    var_bin_list = []
    for col in less_value_features:
        binBadRate = BinBadRate(df2, col, target)[0]
        """
        TODO：感觉有点问题，对于同时存在0% 和 100%的 bad rate 的组，两端都要合并，当前是分别处理的，走两遍流程，有些重复
        """
        if min(binBadRate.values()) == 0:   #由于某个取值没有坏样本而进行合并
            print('{} need to be combined due to 0 bad rate'.format(col))
            # 计算下合并方法
            combine_bin = MergeBad0(df2, col, target)
            merge_bin_dict[col] = combine_bin
            # 按照合并规则，造新变量
            newVar = col + '_Bin'
            df2[newVar] = df2[col].map(combine_bin)
            var_bin_list.append(newVar)
        if max(binBadRate.values()) == 1:   #由于某个取值没有好样本而进行合并
            print('{} need to be combined due to 0 good rate'.format(col))
            # 计算下合并方法
            combine_bin = MergeBad0(df2, col, target, direction='good')
            merge_bin_dict[col] = combine_bin
            # 按照合并规则，造新变量
            newVar = col + '_Bin'
            df2[newVar] = df2[col].map(combine_bin)
            var_bin_list.append(newVar)

    # less_value_features里剩下不需要合并的变量
    less_value_features = [i for i in less_value_features if i + '_Bin' not in var_bin_list]


    # Step 3: 处理取值大于 max_bins 的变量用 bad rate 进行编码，放入连续型变量里进行处理
    br_encoding_dict = {}   #记录按照bad rate进行编码的变量，及编码方式
    br_encoding_features = []
    for col in more_value_features:
        br_encoding = BadRateEncoding(df2, col, target)
        df2[col+'_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']
        br_encoding_features.append(col+'_br_encoding')

    category_info = {
        'less_value_features': less_value_features,
        'less_value_merge_features': var_bin_list,
        'less_value_merge_dict': merge_bin_dict,
        'br_encoding_features': br_encoding_features,
        'br_encoding_dict': br_encoding_dict
    }

    return (df2, category_info)


def ContinuousBinSplit(df, target, num_features, max_bins=5, special_attribute=[]):
    """
    # 针对连续型变量分箱，使用卡方分箱法
    :param df: 包含特征的数据框
    :param target: 目标变量 y
    :param num_features: 数值型变量列表
    :param max_bins: 最大分箱数
    :param special_attribute: 连续型变量特殊取值集合
    :return:
    """

    df2 = df.copy()

    continuous_merged_dict = {}
    var_bin_list = []
    for col in num_features:
        print("{} is in processing".format(col))

        special_attr = [] if -1 not in set(df2[col]) else [-1]
        max_interval = max_bins # 不包含特殊值的情况，有没有特殊值，都是 max_interval，最小为2

        while True:
            cutOff = ChiMerge(df2, col, target, max_interval=max_interval, special_attribute=special_attr, minBinPcnt=0)
            df2[col+'_Bin'] = df2[col].map(lambda x: AssignBin(x, cutOff, special_attribute=special_attr))
            monotone = BadRateMonotone(df2, col+'_Bin', target, special_attribute=special_attr)   # 检验分箱后的单调性是否满足
            if monotone or max_interval == 2:
                break
            else:
                max_interval -= 1

        # cutOff = ChiMerge(trainData, col, 'y', max_interval=max_interval, special_attribute=special_attr, minBinPcnt=0, monotone_check=True)
        newVar = col + '_Bin'
        df2[newVar] = df2[col].map(lambda x: AssignBin(x, cutOff, special_attribute=special_attr))
        var_bin_list.append(newVar)
        continuous_merged_dict[col] = cutOff

    continuous_info = {
        'continuous_merge_features': var_bin_list,
        'continuous_merge_dict': continuous_merged_dict
    }

    return (df2, continuous_info)


def FeaturesForecastImportance(df, target, feature_list, plot_flag=False):
    """
    根据特征分箱结果计算各组WOE值及变量的IV值
    :param df:
    :param target:
    :param feature_list:
    :return:
    """
    WOE_dict = {}
    IV_dict = {}
    for var in feature_list:
        woe_iv = CalcWOE(df, var, target)
        WOE_dict[var] = woe_iv['WOE']
        IV_dict[var] = woe_iv['IV']

    IV = pd.DataFrame(IV_dict, index=['iv']).T
    IV.sort_values(by='iv', ascending=False, inplace=True)

    if plot_flag:
        IV.plot(kind='bar', title='Feature IV')

    return (WOE_dict, IV)


def FeaturesCorrelationAnalysis(df, woe_features, iv_list, corr_threshold=0.7, plot_flag=False):
    """
    根据WOE编码后的特征，分析相关性，在相关性较高特征组中，去除iv值较低的特征
    :param df:
    :param woe_features:
    :return:
    """

    if plot_flag:
        woe_vars = [var + '_WOE' for var in woe_features]
        df2 = df[woe_vars]
        f, ax = plt.subplots(figsize=(10, 8))
        corr = df2.corr()
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax)
        plt.show()

    delete_vars = []
    for var in woe_features:
        new_var = var + '_WOE'
        if var in delete_vars:
            continue
        for var2 in woe_features:
            if var2 == var or var2 in delete_vars:
                continue
            new_var2 = var2 + '_WOE'
            if np.corrcoef(df[new_var], df[new_var2])[0, 1] >= corr_threshold:
                if iv_list.loc[var, 'iv'] > iv_list.loc[var2, 'iv']:
                    delete_vars.append(var2)
                else:
                    delete_vars.append(var)
    clear_woe_features = [var + '_WOE' for var in woe_features if var not in delete_vars]

    return (clear_woe_features, delete_vars)


def FeaturesVIFAnalysis(df, woe_features, plot_flag=False):
    """
    特征集的VIF分析，分析多重共线性

    :param df: 包含需要计算特征的数据集
    :param woe_features: 经过初筛后的变量列表
    :return: 最大的方差扩大因子，方差扩大因子序列
    """

    X = np.matrix(df[woe_features])
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    VIF_df = pd.DataFrame(dict(zip(woe_features, VIF_list)), index=['VIF']).T
    VIF_df.sort_values(by='VIF', ascending=False, inplace=True)

    if plot_flag:
        VIF_df.plot(kind='bar', title='Variance Inflation Factor')
        plt.show()

    return (max(VIF_list), VIF_df)


def FeaturesSelectionRF(df, target, woe_features, top_N=10, plot_flag=False):
    """
    # 使用随机森林方法计算特征重要性

    :param df: 包含特征的数据集
    :param target: 目标变量 y
    :param woe_features: 待筛选的特征列表 X
    :param top_N: 筛选变量的目标个数
    :param plot_flag: 绘图标识，根据特征重要程度绘制 bar 图
    :return: 特征重要程度的 data frame
    """

    X = np.matrix(df[woe_features])
    y = np.array(df[target])

    # 随机森林学习器设定
    RFC = RandomForestClassifier()
    RFC_Model = RFC.fit(X, y)
    feature_importance = {woe_features[i]: RFC_Model.feature_importances_[i] for i in range(len(woe_features))}
    feature_importance = pd.DataFrame(feature_importance, index=['importance']).T
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)

    if plot_flag:
        feature_importance.plot(kind='bar', title='RFC Feature Importance')
        plt.show()

    return feature_importance


def LogisticRegressionModel(train_set, target, features):
    """
    Logistic Regression Model Via 'statsmodels' api instance

    :param train_set: 模型训练数据集
    :param target: 目标变量
    :param features: 进入训练的模型特征
    :return:
    """

    assert set(features) < set(train_set.columns), "Not all vars of feature list in data set!"

    y = train_set[target]
    X = train_set[features]
    X['intercept'] = 1.0

    LR = sm.Logit(y, X).fit()


def StepwiseSelectionFeatures(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    使用 Forward-Backward 方法进行特征选择，按照拟合 Logit 的 p-value 进行

    :param X: pandas.DataFrame 包含候选特征
    :param y: 拟合目标变量
    :param initial_list: 初始特征列表（包含在X.columns中）
    :param threshold_in: include a feature if its p-value < threshold_in
    :param threshold_out: exclude a feature if its p-value > threshold_out
    :param verbose: whether to print the sequence of inclusions and exclusions
    :return: 最终选定的特征列表
    """

    included = list(initial_list)
    while True:
        changed = False
        # Forward Step
        excluded =list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # Backward Step
        model = sm.Logit(y, sm.add_constant(X[included])).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break;
    return included


