#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4 16:45
# @Author  : AndrewMa
# @File    : supervised_bin_split.py

from crsc.bins.bin_utils import BinBadRate


# Supervised Bin Split
# 1. Best-KS
# 2. Chi-Square

def get_best_ks(df_data, start, end, col, total='total', bad='bad', good='good', min_rate=0.05):
    """
    得到当前需求的最佳 KS 切分点

    :param df_data: 要求输入一个变量的频率统计表(要求是按col排序的）， var/total/bad/good
    :param start: 计算起始 index
    :param end: 计算结束 index
    :param col: 变量取值或者分箱结果
    :param total: 箱体样本量
    :param bad: 箱体坏样本量
    :param good: 箱体好样本量
    :param min_rate: 最小分箱占比
    :return:
    """

    # 目标变量列必须在数据集中
    assert {col, total, bad, good} < set(df_data.columns), "Some var not in DataFrame, please check!"

    # 检查起始、结束位置
    nsize = df_data.shape[0]
    assert start >= 0, "Invalid Start Id: %d" % start
    assert end <= nsize, "Invalid End Id: %d" % end
    assert start < end, "Start is bigger than End: (%d, %d)" % (start, end)

    # 核实当前组，是否满足最小箱要求
    total_all = df_data[total].sum()
    df = df_data.loc[start:end].copy()
    N = df[total].sum()
    if N <= total_all * min_rate:  # 检查当前组样本数是否小于最小分箱比例
        raise ValueError("Current bin: (%d, %d) less than min_rate" % (start, end))

    # 检查当前箱，是否满足 B/G 不空要求
    B = df[bad].sum()
    G = df[good].sum()
    if B == 0 or G == 0:  # 检查当前组，是否没有Bad 或者 没有 Good
        raise ValueError("Bad: %d, Good: %d in Current Bin" % (B, G))

    df['bad_pcnt'] = df[bad].apply(lambda x: x * 1.0 / B)
    df['good_pcnt'] = df[good].apply(lambda x: x * 1.0 / G)

    df['bad_pcnt_acc'] = df['bad_pcnt'].cumsum()
    df['good_pcnt_acc'] = df['good_pcnt'].cumsum()
    df['ks'] = df.apply(lambda x: abs(x.bad_pcnt_acc - x.good_pcnt_acc), axis=1)

    ks_list = df['ks'].to_list()
    best_ks_index = ks_list.index(max(ks_list))  # 最佳切分位置
    best_ks_index += start

    # 判断下切分后的情况，是否满足要求
    N_left = df_data.loc[start:best_ks_index, total].sum()
    B_left = df_data.loc[start:best_ks_index, bad].sum()
    N_right = df_data.loc[best_ks_index + 1:end, total].sum()
    B_right = df_data.loc[best_ks_index + 1:end, bad].sum()
    if N_left > total_all * min_rate and N_right > total_all * min_rate:
        if B_left != 0 and B_left != N_left and B_right != 0 and B_right != N_right:
            return best_ks_index
        else:
            print("B=0 or G=0 in SubBins, when split by '%s'" % str(df_data.loc[best_ks_index, col]))
            return -1
    else:
        print("Less than min_rate in SubBins, when split by '%s'" % str(df_data.loc[best_ks_index, col]))
        return -2


def get_ks_split(df_data, col, target, min_rate=0.05):
    """

    :param df_data:
    :param col:
    :param target:
    :param min_rate:
    :return:
    """
    regroup = BinBadRate(df_data, col, target)[1]
    start = 0
    end = regroup.shape[0] - 1
    ks_split = []
    while True:
        ks_id = get_best_ks(regroup, col, 'total', 'bad', 'good', start, end, min_rate)