#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/21 21:34
# @Author  : Andrewma
# @Site    : 
# @File    : mtools.py
# @Software: PyCharm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier


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
        sns.heatmap(corr,
                    mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True,
                    ax=ax)
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

    return clear_woe_features, delete_vars


def FeaturesVIFAnalysis(df, woe_features, plot_flag=False):
    """
    特征集的VIF分析，分析多重共线性

    :param df: 包含需要计算特征的数据集
    :param woe_features: 经过初筛后的变量列表
    :return: 最大的方差扩大因子，方差扩大因子序列
    """

    X = df[woe_features]
    VIF_list = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

    VIF_df = pd.DataFrame(dict(zip(woe_features, VIF_list)), index=['VIF']).T
    VIF_df.sort_values(by='VIF', ascending=False, inplace=True)

    if plot_flag:
        VIF_df.plot(kind='bar', title='Variance Inflation Factor')
        plt.show()

    return max(VIF_list), VIF_df


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

    X = df[woe_features]
    Y = df[target]

    # 随机森林学习器设定
    RFC = RandomForestClassifier()
    RFC_Model = RFC.fit(X, Y)
    feature_importance = {woe_features[i]: RFC_Model.feature_importances_[i] for i in range(len(woe_features))}
    feature_importance = pd.DataFrame(feature_importance, index=['importance']).T
    feature_importance.sort_values(by='importance', ascending=False, inplace=True)

    if plot_flag:
        feature_importance.plot(kind='bar', title='RFC Feature Importance')
        plt.show()

    return feature_importance




