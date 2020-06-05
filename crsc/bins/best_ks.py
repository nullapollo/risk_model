#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:55
# @Author  : AndrewMa
# @File    : best_ks.py

# Best-KS Bin Split

import pandas as pd
import numpy as np
from itertools import combinations


def group_by_df(data, flag_name, factor_name, bad_name='bad', good_name='good', na_trigger=False):
    """
    从 x 和 y 进行分组统计，good/bad 样本

    :param data: 数据 pandas.dataframe
    :param flag_name: y 值列名
    :param factor_name: x 特征列名
    :param bad_name: 命名 bad string
    :param good_name: 命名 good string
    :param na_trigger: 缺失值处理标识
    :return: good/bad 按取值统计的 pandas.dataframe
    """

    if len(data) == 0:
        return pd.DataFrame()

    data = data[flag_name].groupby([data[factor_name], data[flag_name]]).count().unstack().reset_index().fillna(0)
    if len(data.columns) == 3:
        data.columns = [factor_name, good_name, bad_name]  # 重命名列名
        if not na_trigger:  # 非缺失的组，做下类型转换，string 的回 float
            data[factor_name] = data[factor_name].astype(float)
        data = data.sort_values(by=[factor_name], ascending=True)
        data[factor_name] = data[factor_name].astype(str)
        data = data.reset_index(drop=True)
        return data
    else:
        return pd.DataFrame()


def verify_df_two(data_df, flag_name):
    """
    检查是否为二分类问题

    :param data_df: 数据集 pandas.dataframe
    :param flag_name: y 值列名, 默认取值是 0/1
    :return:
    """
    data_df = data_df.dropna(subset=[flag_name])

    check = data_df[data_df[flag_name] > 1]
    if len(check) != 0:
        print('Error: there exits the number bigger than one in the data')
        data_df = pd.DataFrame()
        return data_df

    elif len(data_df) != 0:
        data_df[flag_name] = data_df[flag_name].astype(int)
        return data_df

    else:
        print('Error: the data is wrong')
        data_df = pd.DataFrame()
        return data_df


def get_all_index(var_list, target):
    """
    从目标 list 中，找出击中 target 值的所有索引位置（list.index 只能返回第一个击中的索引位置）

    :param var_list: 搜索列表 list
    :param target: 目标值
    :return:
    """
    # return list(filter(lambda x: var_list[x] == target, range(0, len(var_list))))
    return [idx for idx, value in enumerate(var_list) if value == target]


def best_ks_knot_calculator(data, total_len, good_name, bad_name, start_knot, end_knot, rate):
    """
    Best-KS 单次最优分裂计算

    :param data: 频率统计 pandas.dataframe
    :param total_len: 总体样本数
    :param good_name: good 列名
    :param bad_name: bad 列名
    :param start_knot: 起始位置
    :param end_knot: 结束位置
    :param rate: 最小分箱占比
    :return: 最优分裂位置
    """

    # 只要这个局部的 size
    temp_df = data.loc[start_knot:end_knot]
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])

    # since we'd want to make sure the number of elements in each tiny bin should always greater than 5% of the raw
    # data's length, we need to add some restrictions to the start_knot and the end_knot
    # 从前往后累加，满足最小箱占比的起始分裂位置
    start_add_num = sum(np.cumsum(temp_df[good_name] + temp_df[bad_name]) < rate * total_len)
    # 从后往前累加，满足最小箱占比的最终分裂位置（尾部也要满足）
    end_add_num = sum(np.cumsum(temp_df[good_name] + temp_df[bad_name]) <= temp_len - rate * total_len)

    processed_start_knot = start_knot + start_add_num
    processed_end_knot = start_knot + end_add_num - 1

    if processed_end_knot >= processed_start_knot:
        if sum(temp_df[bad_name]) != 0 and sum(temp_df[good_name]) != 0:  # 当前分裂的都非空
            default_cdf = np.cumsum(temp_df[bad_name]) / sum(temp_df[bad_name])
            undefault_cdf = np.cumsum(temp_df[good_name]) / sum(temp_df[good_name])
            ks_value = max(abs(default_cdf - undefault_cdf).loc[processed_start_knot:processed_end_knot])
            # the index find here is not the final result, we should find the data's position in the outer data set
            index = get_all_index(list(abs(default_cdf - undefault_cdf)), ks_value)
            return temp_df.index[max(index)]
        else:
            return None
    else:
        return None


def best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot, end_knot, current_time):
    """
    Best-KS 分箱核心循环程序，递归

    :param data: 统计数据集 pandas.dataframe
    :param total_len: 全量样本数，用来计算分箱后样本限制条件
    :param max_times: 最大的分裂次数（等于最终想要的分箱个数）
    :param good_name: good 列名
    :param bad_name: bad 列名
    :param rate: 最小箱占比，不低于rate
    :param start_knot: 起始位置
    :param end_knot: 结束位置
    :param current_time: 当前分裂次数
    :return: 分裂点index的列表
    """

    # 取当前需要分箱的子集，求子集的记录数
    temp_df = data.loc[start_knot:end_knot]
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])

    # due to the restriction to the number of elements in the tiny bin
    if temp_len < rate * total_len * 2 or current_time >= max_times:
        return []

    # 核心分裂程序
    new_knot = best_ks_knot_calculator(data, total_len, good_name, bad_name, start_knot, end_knot, rate)
    if new_knot is not None:
        upper_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot,
                                            new_knot, current_time + 1)
        lower_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, new_knot + 1,
                                            end_knot, current_time + 1)
    else:
        upper_result = []
        lower_result = []

    return upper_result + [new_knot] + lower_result


def new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name):
    """
    自动化 Best-KS 分箱程序

    :param data_df: 特征取值的频率统计数据集 pandas.dataframe
    :param total_rec: 全样本量
    :param piece: 目标分箱个数
    :param rate: 最小箱占比
    :param good_name: good 列名
    :param bad_name: bad 列名
    :return: 分裂结果列表
    """

    # call the helper function to finish the following recursion part
    temp_result_list = best_ks_knots_helper(data_df, total_rec, piece, good_name, bad_name, rate, 0, len(data_df), 0)

    split_knot = list(filter(lambda x: x is not None, temp_result_list))
    split_knot.sort()

    return split_knot


def urteil(li):
    """
    检测数值序列的是否满足要求（单调 or u型）

    :param li: 待检测序列 list, 数值型
    :return: 1 满足， 0 不满足
    """
    if len(li) < 4:
        return 1
    else:
        lii = [li[i] - li[i - 1] for i in range(len(li))[1:]]  # 序列做向前差分
        lii = list(map(lambda x: x if x != 0 else 1, lii))
        #        print lii
        zz = np.sign([lii[i] / lii[i - 1] for i in range(len(lii))[1:]]).sum()  # 差分序列向前比较符号
        if zz in [len(li) - 2, len(li) - 4]:  # 单调性, 纯单调是 -2， 一头一尾有一点反差 -4， 其他不认可
            return 1
        else:
            return 0


def iv_calculator(data_df, good_name, bad_name, knots_list, ur=False):
    """
    first, we need to define a IV calculator function.
    The input should be the knots list and the data with unique value's black flag count.
    The result returned by the function will be the IV list of these tiny bins

    :param data_df:
    :param good_name:
    :param bad_name:
    :param knots_list:
    :param ur: 检查 woe 序列是否满足要求（单调 or U型）
    :return:
    """

    # to improve the efficiency of the calculation,
    # I first split the df into a bunch of smaller data frames and put them into a list.
    # Then I Use the map function to do some transformation to calculate the IV value for each small bin
    temp_df_list = []
    for i in range(1, len(knots_list)):
        if i == 1:
            # the data range here we chose to use such format (start, end], since the calculation of CDF is left
            # continuous, thus we need to include the right margin and the left margin should be not included
            # attention: the pd.Series[start:end] is different from pd.DataFrame.loc[start:end]. The previous will not
            # include the end point but the later one will include
            temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
        else:
            temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])

    total_good = sum(data_df[good_name])
    total_bad = sum(data_df[bad_name])
    good_percent_series = pd.Series(list(map(lambda x: float(sum(x[good_name])) / total_good, temp_df_list)))
    bad_percent_series = pd.Series(list(map(lambda x: float(sum(x[bad_name])) / total_bad, temp_df_list)))
    # the woe_list here is used for debugging
    woe_list = list(np.log(bad_percent_series / good_percent_series))
    # here, since we want to make sure the vimp of the result bins is monotonic, thus we add a justification statement
    # here, if it is not monotonic, then it will be discarded and the return will be None
    #    if sorted(woe_list) != woe_list and sorted(woe_list, reverse=True) != woe_list:
    #        return None
    iv_series = (bad_percent_series - good_percent_series) * np.log(bad_percent_series / good_percent_series)
    if np.inf in list(iv_series) or -np.inf in list(iv_series):
        return None

    if ur:  # 检查分箱后计算woe值序列是否满足要求（单调 or U型）
        if urteil(woe_list) == 0:
            return None
        else:
            return sum(iv_series)

    # 更强的要求，必须单调
    if sorted(woe_list) == woe_list or sorted(woe_list, reverse=True) == woe_list:
        return sum(iv_series)
    else:
        return None


def combine_helper(data_df, good_name, bad_name, piece_num, cut_off_list, ur):
    """
    combination_helper function

    :param data_df:
    :param good_name:
    :param bad_name:
    :param piece_num:
    :param cut_off_list: 在分箱数/最小箱占比/单箱同时包含 good & bad 的条件下，按 best-ks 标准得到的切点列表
    :param ur: 检查在特定分箱下，woe序列是否满足 单调 或者 u型
    :return:
    """

    knots_list = list(combinations(cut_off_list, piece_num - 1))  # itertools.combinations 做非重复的排列组合
    # here we do some transformation to the knots list, add the start knot and end knot to all the elements in the list
    # knots_list = map(lambda x: sorted(tuple(set(x + (0, len(data_df) - 1)))), knots_list)
    knots_list = list(map(lambda x: sorted(x + [0, len(data_df) - 1]), knots_list))
    print(len(knots_list))

    # 遍历所有的组合
    iv_for_bins = list(map(lambda x: iv_calculator(data_df, good_name, bad_name, x, ur), knots_list))
    filtered_iv = list(filter(lambda x: x is not None, iv_for_bins))

    if len(filtered_iv) == 0:  # 在该特定分箱数的条件下，没有满足条件（woe序列 单调 或者 单调+U型）的分箱组合
        print('There are no suitable division for the data set with %s pieces' % str(piece_num))
        return None, None
    else:
        if len(get_all_index(iv_for_bins, max(filtered_iv))) > 0:
            target_index = get_all_index(iv_for_bins, max(filtered_iv))[0]  # 对于不唯一的情况，取第一个
            return knots_list[target_index], iv_for_bins[target_index]
        else:
            return None, None


# the cut_off_list here should not contain the start knot 0 and end knot len(data_df)-1, since these knots will be added
# in the later process in function's map part
def combine_tiny_bins0(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    return_piece_num = min(max_piece_num, len(cut_off_list) + 1)
    if return_piece_num == 1:
        return cut_off_list
    for current_piece_num in sorted(range(2, return_piece_num + 1), reverse=True):
        result_knots_list, iv = combine_helper(data_df, good_name, bad_name, current_piece_num, cut_off_list, True)
        if current_piece_num == 2 and iv is not None:
            if iv < 0.1:
                # print("sry, there isn't any suitable division for this data set with the column that you give :(")
                return None
        # here we obey the rule that, the function will return the maximum number of bins with maximum IV value, thus if
        # there is no suitable cut_off_list for the current piece, the number of bins will be minus one
        if result_knots_list is not None:
            return result_knots_list, iv
    #    print("sry, there isn't any suitable division for this data set with the column that you give :(")
    return None


def combine_tiny_bins1(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    """

    :param data_df:
    :param good_name:
    :param bad_name:
    :param max_piece_num:
    :param cut_off_list:
    :return:
    """

    return_piece_num = min(max_piece_num, len(cut_off_list) + 1)
    if return_piece_num == 1:
        return cut_off_list

    # 逆向合并
    for current_piece_num in sorted(range(2, return_piece_num + 1), reverse=True):
        result_knots_list, iv = combine_helper(data_df, good_name, bad_name, current_piece_num, cut_off_list, False)
        if current_piece_num == 2 and iv is not None:  # 2 箱必然单调
            return result_knots_list, iv
        # here we obey the rule that, the function will return the maximum number of bins with maximum IV value, thus if
        # there is no suitable cut_off_list for the current piece, the number of bins will be minus one
        if result_knots_list is not None:
            return result_knots_list, iv
    #    print("sry, there isn't any suitable division for this data set with the column that you give :(")
    return None


# campare two methode of cut: monotonic and U-type.
def combine_tiny_bins(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    #    x0=combine_tiny_bins0(data_df, good_name, bad_name, max_piece_num, cut_off_list)
    x1 = combine_tiny_bins1(data_df, good_name, bad_name, max_piece_num, cut_off_list)
    if x1 is None:
        print("sry, there isn't any suitable division for this data set with the column that you give :(")
        return [0, len(data_df) - 1]
    else:
        return x1[0]


def important_indicator_calculator(data_df, good_name, bad_name, factor_name, knots_list, na_df):
    """
    this function is used for return the import statistical indicators for the binning result

    :param data_df:
    :param good_name:
    :param bad_name:
    :param factor_name:
    :param knots_list:
    :param na_df:
    :return:
    """
    if len(na_df) != 0:
        total_good = sum(data_df[good_name]) + sum(na_df[good_name])
        total_bad = sum(data_df[bad_name]) + sum(na_df[bad_name])
        na_good_percent = na_df[good_name] / float(total_good)
        na_bad_percent = na_df[bad_name] / float(total_bad)

        # 这里单独计算了 缺失/异常值 部分 DataFrame 的 统计特征
        na_indicator = pd.DataFrame(
            {
                'Bin': list(na_df.ix[:, 0]),
                'KS': [None] * len(na_df),
                'WOE': list(np.log(na_bad_percent / na_good_percent)),
                'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                'total_count': list(na_df[good_name] + na_df[bad_name]),
                'bad_rate': list(na_df[bad_name] / (na_df[good_name] + na_df[bad_name]))
            }
        )

    else:  # 没有缺失/异常值的部分
        total_good = sum(data_df[good_name])
        total_bad = sum(data_df[bad_name])
        na_indicator = pd.DataFrame()

    default_cdf = np.cumsum(data_df[bad_name]) / total_bad
    undefault_cdf = np.cumsum(data_df[good_name]) / total_good

    ks_list = list(abs(default_cdf - undefault_cdf).loc[knots_list[:len(knots_list) - 1]])
    temp_df_list = []  # 把 DataFrame 按照分箱切开，其实也可以用分箱生成新的一列，然后重新进行统计，操作更简单点
    bin_list = []
    for i in range(1, len(knots_list)):
        if i == 1:
            temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
            bin_list.append('(-inf, ' + data_df[factor_name][knots_list[i]] + ']')
        else:
            temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])
            if i == len(knots_list) - 1:
                bin_list.append('(' + data_df[factor_name][knots_list[i - 1]] + ', +inf)')
            else:
                bin_list.append(
                    '(' +
                    data_df[factor_name][knots_list[i - 1]] +
                    ', ' +
                    data_df[factor_name][knots_list[i]] +
                    ']'
                )

    good_percent_series = pd.Series(list(map(lambda x: float(sum(x[good_name])) / total_good, temp_df_list)))
    bad_percent_series = pd.Series(list(map(lambda x: float(sum(x[bad_name])) / total_bad, temp_df_list)))
    woe_list = list(np.log(bad_percent_series / good_percent_series))
    iv_list = list((bad_percent_series - good_percent_series) * np.log(bad_percent_series / good_percent_series))
    total_list = list(map(lambda x: sum(x[good_name]) + sum(x[bad_name]), temp_df_list))
    bad_rate_list = list(map(lambda x: float(sum(x[bad_name])) / (sum(x[good_name]) + sum(x[bad_name])), temp_df_list))

    non_na_indicator = pd.DataFrame(
        {
            'Bin': bin_list,
            'KS': ks_list,
            'WOE': woe_list,
            'IV': iv_list,
            'total_count': total_list,
            'bad_rate': bad_rate_list
        }
    )

    result_indicator = pd.concat([non_na_indicator, na_indicator], axis=0).reset_index(drop=True)

    return result_indicator


def all_information(data_df, na_df, total_rec, piece, rate, factor_name, bad_name, good_name, not_in_list):
    """
    整合整个 Best-KS 分箱过程，包括：分裂和合并

    :param data_df: 非空/非异常值部分的统计数据集 pandas.dataframe
    :param na_df: 空值/异常值部分的统计数据集 pandas.dataframe
    :param total_rec: 总体样本数
    :param piece: 分箱个数
    :param rate: 最小箱占比
    :param factor_name: 特征列名称
    :param bad_name: bad 列名
    :param good_name: good 列名
    :param not_in_list: 空值/异常值 列表
    :return:
    """

    # =========================================================================
    # Step 1: Best-KS 分裂（只对非空/非异常的数据集部分）
    # =========================================================================
    split_knots = new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name)
    print(split_knots)

    # =========================================================================
    # Step 2: 遍历合并（在最大可分裂下，按照 woe 单调性以及 iv 值找最优）
    # =========================================================================
    best_knots = combine_tiny_bins(data_df, good_name, bad_name, piece, split_knots)

    if best_knots is None and (min(data_df[bad_name]) > 0 and min(data_df[good_name]) > 0):
        na_df = na_df.append(data_df)
        not_in_list = not_in_list + list(data_df.ix[:, 0])

    result_indicator = important_indicator_calculator(data_df, good_name, bad_name, factor_name, best_knots, na_df)

    return result_indicator, not_in_list, best_knots


def best_ks_bin(flag_name, factor_name, data=pd.DataFrame(), bad_name='bad', good_name='good', piece=5, rate=0.05,
                min_bin_size=50, not_in_list=None):
    """
    Best-KS 分箱 API

    :param flag_name: y 值列名
    :param factor_name: x 列名
    :param data: 数据集 pandas.dataframe
    :param bad_name: bad 列名
    :param good_name: good 列名
    :param piece: 分箱数
    :param rate: 最小箱样本占总体的比值
    :param min_bin_size: 最小箱至少有 min_bin_size 个样本
    :param not_in_list: 缺失或者异常值的列表
    :return:
    """

    if not_in_list is None:
        not_in_list = []
    if len(data) == 0:
        print('Error: there is no data')
        return pd.DataFrame()

    work_data = data.loc[data.index, [factor_name, flag_name]]  # 只要 x 和 y 列
    work_data = verify_df_two(work_data, flag_name)  # 检查是否为二分类
    if len(work_data) == 0:  # 检查有误的，直接返回退出
        return pd.DataFrame

    # after that, we want to separate the current df into two parts, NA and non-NA
    # the very first thing here should be transforming the type of value in factor_name column into str type
    work_data[factor_name] = work_data[factor_name].astype(str)
    # since there will be the None and nan be transformed into the str type, thus we need to add some default value into
    # the not_in_list, the set here may be redundant
    not_in_list = not_in_list + ['None', 'nan']
    na_df = work_data.loc[work_data[factor_name].apply(lambda x: x in not_in_list)]
    non_na_df = work_data.loc[work_data[factor_name].apply(lambda x: x not in not_in_list)]

    # generate the grouped_by format which is used for the later process
    na_df = group_by_df(na_df, flag_name, factor_name, bad_name, good_name, True)
    non_na_df = group_by_df(non_na_df, flag_name, factor_name, bad_name, good_name, False)
    #    print factor_name
    if len(non_na_df) == 0:
        print('sry, there are no data available for separate process :(')
        return pd.DataFrame(), not_in_list
    # total_good = sum(non_na_df[good_name]) + sum(na_df[good_name])
    # total_bad = sum(non_na_df[bad_name]) + sum(na_df[bad_name])
    # total_rec = total_good + total_bad
    total_rec = work_data.shape[0]  # 总样本数
    min_bin_size_r = max(rate, min_bin_size / float(total_rec))
    result, not_in_list, best_knots = all_information(non_na_df, na_df, total_rec, piece, min_bin_size_r, factor_name,
                                                      bad_name, good_name, not_in_list)
    # print(time.localtime(time.time()))
    if len(best_knots) == 2:
        print('sry, there are no data available for separate process :(')
        return pd.DataFrame(), not_in_list

    return result, not_in_list
