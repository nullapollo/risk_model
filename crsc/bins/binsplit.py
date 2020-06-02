#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 9:51
# @Author  : AndrewMa
# @File    : binsplit.py

from crsc.bins.binutils import *


def CategoryBinSplit(df, target, cat_features, max_bins=5):
    """
    类别变量进行特征分箱处理

    是否有超过 max_bin 数的取值
    1. 是：采用各取值的badrate进行编码，然后调用连续型的分箱算法
    2. 否： 是否有需要合并的（0% 或者 100%）的情况
            1. 是： 进行合并，按照 bad rate 排序，往中间进行合并，0 找最小 非0 ，1 找最大非 1
            2. 否： 直接输出

    :param df: 包含变量的数据集 pandas.dataframe
    :param target: 分类标签 Y, 0/1 数值
    :param cat_features: 分类变量列表 list
    :return: 新增分箱特征的数据集df2, 分类变量信息字典
    """

    df2 = df.copy()
    more_value_features = []
    less_value_features = []

    # Step 1：先处理分类变量，检查类别型变量中，哪些变量取值超过 max_bins
    for var in cat_features:
        valueCounts = len(set(df2[var]))  # 取值个数
        print(var, valueCounts)
        if valueCounts > max_bins:
            more_value_features.append(var)  # 取值超过 max_bins 的变量，需要 bad rate编码，再用卡方分箱法进行分箱
        else:
            less_value_features.append(var)

    # Step 2: 处理水平少于 max_bins 的变量，只用检查是否同时存在正负样本
    merge_bin_dict = {}
    var_bin_list = []
    for col in less_value_features:
        binBadRate = BinBadRate(df2, col, target)[0]  # 字典类型 key-value, 取值 - bad rate
        # TODO: 对于同时存在 纯0 和 纯1 的组，这里目前没有处理到位
        if min(binBadRate.values()) == 0:  # 由于某个取值没有坏样本而进行合并
            print("'%s' need to be combined due to 0 bad rate" % col)
            # 计算下合并方法
            combine_bin = MergeBad0(df2, col, target)  # 返回也是个字典 取值 - 箱id

            # 保存下分箱的 Map
            merge_bin_dict[col] = combine_bin  # 字典的字典 {变量 - {取值 - 箱 id}}

            # 按照合并规则，造新变量
            newVar = col + '_Bin'
            df2[newVar] = df2[col].map(combine_bin)
            var_bin_list.append(newVar)

        if max(binBadRate.values()) == 1:  # 由于某个取值没有好样本而进行合并
            print("'%s' need to be combined due to 0 good rate" % col)
            # 计算下合并方法
            combine_bin = MergeBad0(df2, col, target, direction='good')

            merge_bin_dict[col] = combine_bin

            # 按照合并规则，造新变量
            newVar = col + '_Bin'
            df2[newVar] = df2[col].map(combine_bin)
            var_bin_list.append(newVar)

    # less_value_features里剩下不需要合并的变量
    less_value_features = [v for v in less_value_features if v + '_Bin' not in var_bin_list]

    # Step 3: 处理取值大于 max_bins 的变量用 bad rate 进行编码，放入连续型变量里进行处理
    br_encoding_dict = {}  # 记录按照bad rate进行编码的变量，及编码方式
    br_encoding_features = []
    for col in more_value_features:
        br_encoding = BadRateEncoding(df2, col, target)
        df2[col + '_br_encoding'] = br_encoding['encoding']
        br_encoding_dict[col] = br_encoding['bad_rate']  # 保存 bad rate 编码的 Map
        br_encoding_features.append(col + '_br_encoding')

    category_info = {
        'less_value_features': less_value_features,  # 取值较少的, 不用合并的
        'less_value_merge_features': var_bin_list,  # 取值较少，经过合并的
        'less_value_merge_dict': merge_bin_dict,  # 合并的 Map

        'br_encoding_features': br_encoding_features,  # 取值较多，需要 bad rate 编码的
        'br_encoding_dict': br_encoding_dict  # 去交较多，编码的 Map
    }

    return df2, category_info


def ContinuousBinSplit(df, target, num_features, max_bins=5, special_attribute=[]):
    """
    连续型变量分箱，使用卡方分箱法

    :param df: 包含特征的数据框 pandas.dataframe
    :param target: 目标变量 Y 0/1 取值，非字符
    :param num_features: 数值型变量列表 list
    :param max_bins: 最大分箱数
    :param special_attribute: 连续型变量特殊取值集合 list
    :return:
    """

    df2 = df.copy()

    continuous_merged_dict = {}
    var_bin_list = []
    for col in num_features:
        print("'%s' is in processing" % col)

        special_attr = [] if -1 not in set(df2[col]) else [-1]
        max_interval = max_bins  # 不包含特殊值的情况，有没有特殊值，都是 max_interval，最小为2

        while True:
            cutOff = ChiMerge(df2, col, target, max_interval=max_interval, special_attribute=special_attr, minBinPcnt=0)
            df2[col + '_Bin'] = df2[col].map(lambda x: AssignBin(x, cutOff, special_attribute=special_attr))
            # 检验分箱后, 每箱中的坏样本率（总体占比），是否满足单调性
            monotone = BadRateMonotone(df2, col + '_Bin', target, special_attribute=special_attr)

            if monotone or max_interval == 2:
                break
            else:  # 不满足单调性的时候，需要重新做
                max_interval -= 1

        newVar = col + '_Bin'
        # df2[newVar] = df2[col].map(lambda x: AssignBin(x, cutOff, special_attribute=special_attr))
        var_bin_list.append(newVar)
        continuous_merged_dict[col] = cutOff

    continuous_info = {
        'continuous_merge_features': var_bin_list,  # 被分箱的连续变量
        'continuous_merge_dict': continuous_merged_dict  # 分箱的切分点
    }

    return df2, continuous_info
