#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 11:26
# @Author  : AndrewMa
# @File    : woe.py

import numpy as np
import pandas as pd


def CalcWOE(df, col, target):
    """
    根据变量分箱，计算变量的 woe 和 iv 值

    :param df: 包含需要计算WOE的变量和目标变量
    :param col: 需要计算WOE、IV的变量，必须是分箱后的变量，或者不需要分箱的类别型变量
    :param target: 目标变量，0、1表示好、坏
    :return: 返回WOE和IV
    """
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})

    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)  # level=0， 会把原来groupby 的 index 保留下来当 col
    regroup['good'] = regroup['total'] - regroup['bad']

    N = regroup['total'].sum()
    B = regroup['bad'].sum()
    G = N - B

    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['woe'] = regroup.apply(lambda x: np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)
    regroup['iv'] = regroup.apply(lambda x: (x.bad_pcnt - x.good_pcnt) * x.woe, axis=1)

    WOE_dict = dict(zip(regroup.col, regroup.woe))
    IV = regroup['iv'].sum()

    return {"WOE": WOE_dict, 'IV': IV}


def FeaturesForecastImportance(df, target, feature_list, plot_flag=False):
    """
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
