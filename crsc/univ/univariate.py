#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/28 22:36
# @Author  : Andrew Ma
# @Site    : 
# @File    : univariate.py
# @Software: PyCharm


import pandas as pd


def feature_check(df_data, cols, missing_id='\\N'):
    """
    数据摸底，检查特征的缺失率与取值的水平数

    :param df_data: 包含特征的数据集 pandas.dataframe
    :param cols: 待检查的变量列表 list
    :param missing_id: 缺失值标志
    :return: 变量的缺失率和取值水平数 pandas.dataframe
    """
    assert set(cols) < set(df_data.columns), "Not all vars of feature list in data set!"

    missing_stats = {}
    vars_levels = {}  # 非缺失的取值个数（水平数）

    nobs = df_data.shape[0]
    for var in cols:
        missing_stats[var] = df_data[var].map(lambda x: 1. if x == missing_id else 0.).sum() / nobs
        vars_levels[var] = len(set(df_data.loc[df_data[var] != missing_id][var]))
    df_data_missing = pd.DataFrame(missing_stats, index=['missing']).T
    df_data_levels = pd.DataFrame(vars_levels, index=['levels']).T

    vars_info = pd.merge(left=df_data_missing, right=df_data_levels, left_index=True, right_index=True)
    vars_info.sort_values(by='missing', ascending=False, inplace=True)

    return vars_info


def con_vars_dist(df_data, cols):
    """
    检查连续型变量的数据分布，分位数检查, 离散型变量直接 value_counts

    :param df_data: 包含连续特征的数据集 pandas.dataframe
    :param cols: 连续特征变量列表 list
    :return: 各连续变量分位数取值 pandas.dataframe
    """
    assert set(cols) < set(df_data.columns), "Not all vars of feature list in data set!"

    dist_df = df_data[cols].quantile([x / 10. for x in range(0, 11)]).T
    dist_df.columns = ['Min'] + ['Q' + str(x) for x in [x * 10 for x in range(1, 10)]] + ['Max']

    return dist_df
