#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/28 22:36
# @Author  : Andrew Ma
# @Site    : 
# @File    : scorecard_dp.py
# @Software: PyCharm


import numpy as np
import pandas as pd


def feature_check(df, cols, missing_id='\\N'):
    """
    数据摸底，检查特征的缺失率与取值的水平数

    :param df: 包含特征的数据集
    :param cols: 待检查的数据集列表
    :param missing_id: 缺失值标志
    :return: 变量的缺失率和取值水平数的数据框
    """
    assert set(cols) < set(df.columns), "Not all vars of feature list in data set!"

    missing_stats = {}
    vars_levels = {}

    nobs = df.shape[0]
    for var in cols:
        missing_stats[var] = df[var].map(lambda x: 1. if x == missing_id else 0.).sum() / nobs
        vars_levels[var] = len(set(df.loc[df[var] != missing_id][var]))
    df_missing = pd.DataFrame(missing_stats, index=['missing']).T
    df_levels = pd.DataFrame(vars_levels, index=['levels']).T

    vars_info = pd.merge(left=df_missing, right=df_levels, left_index=True, right_index=True)
    vars_info.sort_values(by='missing', ascending=False, inplace=True)

    return vars_info


def con_vars_dist(df, cols):
    """
    检查连续型变量的数据分布，分位数检查

    :param df: 包含连续特征的数据集
    :param cols: 连续特征变量列表
    :return: 各连续变量分位数取值 DataFrame
    """
    assert set(cols) < set(df.columns), "Not all vars of feature list in data set!"

    dist_df = df[cols].quantile([x / 10. for x in range(0, 11)]).T
    dist_df.columns = ['min'] + ['q' + str(x) for x in [x * 10 for x in range(1, 10)]] + ['max']

    return dist_df































