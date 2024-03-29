#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/4 19:33
# @Author  : AndrewMa
# @File    : tmp.py
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import missingno as msno
plt.style.use('fivethirtyeight')
import warnings
import datetime

warnings.filterwarnings('ignore')
# %matplotlib inline
from tqdm import tqdm

import re
import math
import time
import itertools
import random

from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import os


#######################################################KS分箱的主体逻辑##############################################
# def init_logger(logger_name, logging_path):
#     if not os.path.exists(logging_path):
#         os.makedirs(logging_path)
#     if logger_name not in Logger.manager.loggerDict:
#         logger = logging.getLogger(logger_name)
#         logger.setLevel(logging.DEBUG)
#         handler = TimedRotatingFileHandler(filename=logging_path + "/%sAll.log" % logger_name, when='D', backupCount=7)
#         datefmt = '%Y-%m-%d %H:%M:%S'
#         format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
#         formatter = logging.Formatter(format_str, datefmt)
#         handler.setFormatter(formatter)
#         handler.setLevel(logging.INFO)
#         logger.addHandler(handler)
#         console = logging.StreamHandler()
#         console.setLevel(logging.INFO)
#         console.setFormatter(formatter)
#         logger.addHandler(console)
#         handler = TimedRotatingFileHandler(filename=logging_path + "/%sError.log" % logger_name, when='D',
#                                            backupCount=7)
#         datefmt = '%Y-%m-%d %H:%M:%S'
#         format_str = '[%(asctime)s]: %(name)s %(filename)s[line:%(lineno)s] %(levelname)s  %(message)s'
#         formatter = logging.Formatter(format_str, datefmt)
#         handler.setFormatter(formatter)
#         handler.setLevel(logging.ERROR)
#         logger.addHandler(handler)
#     logger = logging.getLogger(logger_name)
#     return logger


def get_max_ks(date_df, start, end, rate, factor_name, bad_name, good_name, total_name, total_all):
    '''
    计算最大的ks值
    :param date_df: 数据源
    :param start: 第一条数据的index
    :param end: 最后一条数据的index
    :param rate:
    :param factor_name:
    :param bad_name:
    :param good_name:
    :param total_name:
    :param total_all:
    :return:最大ks值切点的index
    '''
    ks = ''
    # 获取黑名单数据
    bad = date_df.loc[start:end, bad_name]
    # 获取白名单数据
    good = date_df.loc[start:end, good_name]

    # np.cumsum累加。计算黑白的数量占比，累计差
    bad_good_cum = list(abs(np.cumsum(bad / sum(bad)) - np.cumsum(good / sum(good))))
    if bad_good_cum:
        # 找到最大的ks
        max_ks = max(bad_good_cum)
        # 找到最大ks的切点index。
        index_max = bad_good_cum.index(max_ks)
        t = start + index_max
        len1 = sum(date_df.loc[start:t, total_name])
        len2 = sum(date_df.loc[t + 1:end, total_name])
        # 这个就是rate起的效果，一旦按照最大ks切点切割数据，要保证两边的数据量都不能小于一个阈值
        if len1 >= rate * total_all:
            if len2 >= rate * total_all:
                ks = t
    # 如果分割之后，任意一部分数据的数量小于rate这个阈值，那么ks就返回为空了。
    return ks


def cut_fun(x, date_df, types, rate, factor_name, bad_name, good_name, total_name, total_all):
    '''

    :param x: List，就是保存了date_df的第一条index和最后一条index的List。
    :param date_df: 数据源
    :param types: 不知道是什么意思
    :param rate: rate的含义也是一直不清楚
    :param factor_name: 待分箱的特征字段
    :param bad_name:
    :param good_name:
    :param total_name:
    :param total_all:
    :return: 数据的start index,切点index,end index。
    '''
    if types == 'upper':
        # 起始从date_df的第一条开始
        start = x[0]
    else:
        start = x[0] + 1
    # 结束时date_df的最后一条
    end = x[1]
    t = ''
    # 很明显start != end,所以就执行这个函数体
    if start != end:
        # 计算得到最大ks切点index的值，并且把值存入t。
        t = get_max_ks(date_df, start, end, rate, factor_name, bad_name, good_name, total_name, total_all)
    if t:
        # 把t存入x。
        x.append(t)
        # 这个时候x存着[start，切点，end]
        x.sort()
    if t == 0:
        x.append(t)
        x.sort()

    return x


def cut_while_fun(t_list, date_df, rate, factor_name, bad_name, good_name, total_name, total_all):
    '''

    :param t_list: start_index,分箱切点 ,end_index
    :param date_df:
    :param rate:
    :param factor_name:
    :param bad_name:
    :param good_name:
    :param total_name:
    :param total_all:
    :return:
    '''
    if len(t_list) != 2:
        # 切点左边数据
        t_up = [t_list[0], t_list[1]]
        # 切点右边数据
        t_down = [t_list[1], t_list[2]]

        # 递归对左边数据进行切割
        if t_list[1] - t_list[0] > 1 and sum(date_df.loc[t_up[0]:t_up[1], total_name]) >= rate * sum(
                date_df[total_name]):

            t_up = cut_fun(t_up, date_df, 'upper', rate, factor_name, good_name, bad_name, total_name, total_all)
        else:
            t_up = []

        # 递归对右边数据进行切割
        if t_list[2] - t_list[1] > 1 and sum(date_df.loc[t_down[0] + 1:t_down[1], total_name]) >= rate * sum(
                date_df[total_name]):
            t_down = cut_fun(t_down, date_df, 'down', rate, factor_name, good_name, bad_name, total_name, total_all)
        else:
            t_down = []
    else:
        t_up = []
        t_down = []
    return t_up, t_down


def ks_auto(date_df, piece, rate, factor_name, bad_name, good_name, total_name, total_all):
    '''
    :param date_df: 数据源
    :param piece: 分箱数目
    :param rate: 最小数量占比，就是把数据通过切点分成两半部分之后，要保证两部分的数量都必须不能小于这个占比rate。
    :param factor_name: 待分箱的特征名称
    :param bad_name: 黑名单特征名称
    :param good_name: 白名单特征名称
    :param total_name: 总和的特诊名称
    :param total_all: 总共数据量
    :return: 返回整个分箱的间隔点，用List保存。这里是以date_df的index为分割点的。
    '''
    t1 = 0
    # 数据源的大小，条数
    t2 = len(date_df) - 1
    num = len(date_df)
    # 还不知道这样做的目的是什么。
    if num > pow(2, piece - 1):
        num = pow(2, piece - 1)

    # 新定义一个list,这个list是什么含义
    t_list = [t1, t2]
    tt = []
    i = 1
    # 如果数据源的条数大于1，就表示有分箱的资格
    if len(date_df) > 1:
        # 这个是为了获取date_df数据的[start_index，切点_index, end_index]
        # 将数据根据ks最大处进行二分
        t_list = cut_fun(t_list, date_df, 'upper', rate, factor_name, bad_name, good_name, total_name, total_all)
        tt.append(t_list)
        for t_new in tt:
            # >2说明，分箱是成功的。
            if len(t_new) > 2:
                #
                up_down = cut_while_fun(t_new, date_df, rate, factor_name, bad_name, good_name, total_name, total_all)
                t_up = up_down[0]
                if len(t_up) > 2:
                    #
                    t_list = list(set(t_list + t_up))
                    tt.append(t_up)
                t_down = up_down[1]
                if len(t_down) > 2:
                    t_list = list(set(t_list + t_down))
                    tt.append(t_down)
                i += 1
                # 注意循环的停止条件
                # 1. i表示通过箱数限制break
                # 2. len(t_list)还不是很清楚
                if len(t_list) - 1 > num:
                    break
                if i >= piece:
                    break
    if len(date_df) > 0:
        # 这里有个疑问，我感觉有问题
        # 这里为啥要获取第一条数据，total的数量
        length1 = date_df.loc[0, total_name]
        if length1 >= rate * total_all:
            if 0 not in t_list:
                t_list.append(0)
        else:
            t_list.remove(0)
    t_list.sort()
    return t_list


def get_combine(t_list, date_df, piece):
    '''
    :param t_list: 这个值分箱间隔点
    :param date_df: 数据源
    :param piece: 分箱的箱数，表示第几箱。
    :return: 枚举所有的分箱可能组合
    '''
    t1 = 0
    t2 = len(date_df) - 1
    list0 = t_list[1:len(t_list) - 1]
    combine = []
    if len(t_list) - 2 < piece:
        c = len(t_list) - 2
    else:
        c = piece - 1
    # 获取list0的所有子序列。子序列长度是c
    list1 = list(itertools.combinations(list0, c))
    if list1:
        # 向list1收尾添加数据，头部添加t1-1,尾部添加t2
        combine = map(lambda x: sorted(x + (t1 - 1, t2)), list1)
    return combine


def cal_iv(date_df, items, bad_name, good_name, total_name):
    '''

    :param date_df:
    :param items:
    :param bad_name:
    :param good_name:
    :param total_name:
    :return: 返回计算的IV值
    '''
    iv0 = 0
    bad0 = np.array(map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items))
    good0 = np.array(map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items))
    bad_rate0 = np.array(
        map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]) * 1.0 / sum(date_df.ix[x[0]:x[1], total_name]), items))
    if 0 in bad0:
        return iv0
    if 0 in good0:
        return iv0
    good_per0 = good0 * 1.0 / sum(date_df[good_name])
    bad_per0 = bad0 * 1.0 / sum(date_df[bad_name])
    woe0 = map(lambda x: math.log(x, math.e), good_per0 / bad_per0)
    if sorted(woe0, reverse=False) == list(woe0) and sorted(bad_rate0, reverse=True) == list(bad_rate0):
        iv0 = sum(woe0 * (good_per0 - bad_per0))
    elif sorted(woe0, reverse=True) == list(woe0) and sorted(bad_rate0, reverse=False) == list(bad_rate0):
        iv0 = sum(woe0 * (good_per0 - bad_per0))
    return iv0


def choose_best_combine(date_df, combine, bad_name, good_name, total_name):
    '''
    :param date_df: 数据源
    :param combine: 所有的分箱可能
    :param bad_name:
    :param good_name:
    :param total_name:
    :return: 通过最大IV值，来得到最优的分箱方法
    '''
    z = [0] * len(combine)
    for i in range(len(combine)):
        item = combine[i]
        z[i] = (zip(map(lambda x: x + 1, item[0:len(item) - 1]), item[1:]))
    # 计算最大的IV值
    iv_list = map(lambda x: cal_iv(date_df, x, bad_name, good_name, total_name), z)
    iv_max = max(iv_list)
    if iv_max == 0:
        return ''
    index_max = iv_list.index(iv_max)
    combine_max = z[index_max]
    # 返回最好的分箱组合

    # [(0, 180), (181, 268), (269, 348), (349, 450), (451, 605)] 类似于这种数据

    return combine_max


def verify_woe(x):
    if re.match('^\d*\.?\d+$', str(x)):
        return x
    else:
        return 0


def best_df(date_df, items, na_df, rate, factor_name, total_name, bad_name, good_name, total_all, good_all, bad_all):
    '''

    :param date_df:
    :param items: 分箱间隔，数组[(0, 180), (181, 268), (269, 348), (349, 450), (451, 605)]
    :param na_df:
    :param rate:
    :param factor_name:
    :param total_name:
    :param bad_name:
    :param good_name:
    :param total_all:
    :param good_all:
    :param bad_all:
    :return:分箱之后的指标保存为dataframe，并返回。
    '''
    df0 = pd.DataFrame()

    if items:
        piece0 = map(
            lambda x: '[' + str(date_df.ix[x[0], factor_name]) + ',' + str(date_df.ix[x[1], factor_name]) + ']', items)
        bad0 = map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items)
        good0 = map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items)

        if len(na_df) > 0:
            piece0 = np.array(list(piece0) + map(lambda x: '[' + str(x) + ',' + str(x) + ']', list(na_df[factor_name])))
            bad0 = np.array(list(bad0) + list(na_df[bad_name]))
            good0 = np.array(list(good0) + list(na_df[good_name]))
        else:
            piece0 = np.array(list(piece0))
            bad0 = np.array(list(bad0))
            good0 = np.array(list(good0))

        # bad0,good0都是list数据结构
        total0 = bad0 + good0
        # 计算每一个箱子的总数量占比
        total_per0 = total0 * 1.0 / total_all
        # 当前箱子的黑名单比例
        bad_rate0 = bad0 * 1.0 / total0
        # 当前箱子的白名单比例
        good_rate0 = 1 - bad_rate0
        # 当前箱子的白名单在整体白名单数据的比例
        good_per0 = good0 * 1.0 / good_all
        # 当前箱子黑名单在在整体黑名单数据的比例
        bad_per0 = bad0 * 1.0 / bad_all
        # 先将这些数据保存为数框
        df0 = pd.DataFrame(zip(piece0, total0, bad0, good0, total_per0, bad_rate0, good_rate0, good_per0, bad_per0),
                           columns=['Bin', 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate',
                                    'Good_Pcnt', 'Bad_Pcnt'])
        # 通过bad_rate进行排序
        df0 = df0.sort_values(by='Bad_Rate', ascending=False)
        df0.index = range(len(df0))
        bad_per0 = np.array(list(df0['Bad_Pcnt']))
        good_per0 = np.array(list(df0['Good_Pcnt']))
        bad_rate0 = np.array(list(df0['Bad_Rate']))
        good_rate0 = np.array(list(df0['Good_Rate']))
        bad_cum = np.cumsum(bad_per0)
        good_cum = np.cumsum(good_per0)
        #
        woe0 = map(lambda x: math.log(x, math.e), good_per0 / bad_per0)
        # 这里要注意当woe是无穷大的情况
        # 这种情况是因为在某些箱体中，黑名单数量或者白名单数量为0造成的
        if 'inf' in str(woe0):
            woe0 = map(lambda x: verify_woe(x), woe0)
        iv0 = woe0 * (good_per0 - bad_per0)
        gini = 1 - pow(good_rate0, 2) - pow(bad_rate0, 2)
        df0['Bad_Cum'] = bad_cum
        df0['Good_Cum'] = good_cum
        df0["Woe"] = woe0
        df0["IV"] = iv0
        df0['Gini'] = gini
        # 就是累计到KS最大的那个点
        df0['KS'] = abs(df0['Good_Cum'] - df0['Bad_Cum'])
    # 返回数框
    return df0


def all_information(date_df, na_df, piece, rate, factor_name, total_name, bad_name, good_name, total_all, good_all,
                    bad_all):
    '''

    :param date_df: 这是经过处理之后的数据源，主要是针对factor_name统计flag_name的good,bad数量的数据
    :param na_df:   这是个空的df。
    :param piece:  分片大小，就是箱数
    :param rate: 值是0.05,这个值目前的含义不明
    :param factor_name:  分箱特征
    :param total_name:  总和的特征名称
    :param bad_name:   黑名单的特征名称
    :param good_name:  白名单的特征名称
    :param total_all:  总和数量
    :param good_all: 白名单数量
    :param bad_all:  黑名单数量
    :return:分箱之后的所有结果
    '''
    # 新创建的一个List
    p_sort = range(piece + 1)
    # 倒着排序，就是从大到小排序
    p_sort.sort(reverse=True)

    t_list = ks_auto(date_df, piece, rate, factor_name, bad_name, good_name, total_name, total_all)

    # 就是说明不需要分箱
    if len(t_list) < 3:
        df1 = pd.DataFrame()
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1
    df1 = pd.DataFrame()
    for c in p_sort[:piece - 1]:
        # 枚举所有的分箱可能组合。
        combine = get_combine(t_list, date_df, c)

        # 选出最好的分箱
        best_combine = choose_best_combine(date_df, combine, bad_name, good_name, total_name)
        # 按照最佳的分箱数组，分箱
        df1 = best_df(date_df, best_combine, na_df, rate, factor_name, total_name, bad_name, good_name, total_all,
                      good_all, bad_all)
        if len(df1) != 0:
            gini = sum(df1['Gini'] * df1['Total_Num'] / sum(df1['Total_Num']))
            print
            'piece_count:', str(len(df1))
            print
            'IV_All_Max:', str(sum(df1['IV']))
            print
            'Best_KS:', str(max(df1['KS']))
            print
            'Gini_index:', str(gini)
            print
            df1
            # 把分箱之后的各个指标存为df，并且返回。
            return df1
    if len(df1) == 0:
        raise Warning('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1


def fun_group_by(date_df, factor_name, bad_name, good_name):
    df_bad = date_df.groupby(factor_name)[bad_name].agg([(bad_name, 'sum')])
    df_good = date_df.groupby(factor_name)[good_name].agg([(good_name, 'sum')])
    df_bad = df_bad.reset_index()
    df_good = df_good.reset_index()
    good_dict = dict(zip(list(df_good[factor_name]), list(df_good[good_name])))
    df_bad[good_name] = df_bad[factor_name].map(good_dict)
    df_bad[factor_name] = df_bad[factor_name].apply(lambda x: verify_factor(x))
    df_bad = df_bad.sort_values(by=[factor_name], ascending=True)
    df_bad[factor_name] = df_bad[factor_name].astype(str)
    return df_bad


def verify_factor(x):
    '''

    :param x:
    :return:
    '''
    if re.match('^\-?\d*\.?\d+$', x):
        x = float(x)
    return x


def path_df(path, sep, factor_name):
    data = pd.read_csv(path, sep=sep)
    data[factor_name] = data[factor_name].astype(str).map(lambda x: x.upper())
    data[factor_name] = data[factor_name].apply(lambda x: re.sub(' ', 'MISSING', x))
    return data


def verify_df_multiple(date_df, factor_name, total_name, bad_name, good_name):
    date_df = date_df.fillna(0)
    if (bad_name in date_df.columns) & (good_name in date_df.columns) & (total_name not in date_df.columns):
        date_df[good_name] = date_df[good_name].astype(float)
        date_df[bad_name] = date_df[bad_name].astype(float)
        date_df[total_name] = date_df[bad_name] + date_df[good_name]
        date_df = date_df.drop(date_df[date_df[total_name] == 0].index)
    if total_name in date_df.columns:
        date_df = date_df.drop(date_df[date_df[total_name] == 0].index)
        if bad_name in date_df.columns and good_name in date_df.columns:
            date_df['check'] = date_df[good_name] + date_df[bad_name] - date_df[total_name]
            date_df_check = date_df[date_df['check'] != 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print
                'Error: total amounts is not equal to the sum of bad & good amounts'
                print
                date_df_check
        elif bad_name in date_df.columns:
            date_df['check'] = date_df[total_name] - date_df[bad_name]
            date_df_check = date_df[date_df['check'] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print
                'Error: total amounts is smaller than bad amounts'
                print
                date_df_check
            else:
                date_df[good_name] = date_df[total_name] - date_df[bad_name]
        elif good_name in date_df.columns:
            date_df['check'] = date_df[total_name] - date_df[good_name]
            date_df_check = date_df[date_df['check'] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print
                'Error: total amounts is smaller than good amounts'
                print
                date_df_check
            else:
                date_df[bad_name] = date_df[total_name] - date_df[good_name]
        else:
            print
            'Error: lack of bad or good data'
            date_df = pd.DataFrame()
    elif bad_name not in date_df.columns:
        print
        'Error: lack of bad data'
        date_df = pd.DataFrame()
    elif good_name not in date_df.columns:
        print
        'Error: lack of good data'
        date_df = pd.DataFrame()
    if len(date_df) != 0:
        date_df[good_name] = date_df[good_name].astype(int)
        date_df[bad_name] = date_df[bad_name].astype(int)
        date_df[factor_name] = date_df[factor_name].apply(lambda x: verify_factor(x))
        date_df = date_df.sort_values(by=[factor_name], ascending=True)
        date_df[factor_name] = date_df[factor_name].astype(str)
        del date_df['check']
    return date_df


def verify_df_two(date_df, flag_name, factor_name):
    '''
    验证数据集
    :param date_df:
    :param flag_name:
    :param factor_name:
    :return:
    '''
    # 先删除flag_name为空的数据
    date_df = date_df.drop(date_df[date_df[flag_name].isnull()].index)
    # 获取flag_name值大于1的数据。如果是二分类，flag_name只会是0和1，不应该出现大于1的情况。
    check = date_df[date_df[flag_name] > 1]
    if len(check) != 0:
        print
        'Error: there exits the number bigger than one in the data'
        date_df = pd.DataFrame()
        return date_df
    elif len(date_df) != 0:
        # 这是正常，说明是二分类问题，并且转化flag_name的值为int类型。
        date_df[flag_name] = date_df[flag_name].astype(int)
        return date_df
    else:
        print
        'Error: the data is wrong'
        date_df = pd.DataFrame()
        return date_df


def universal_df(data, flag_name, factor_name, total_name, bad_name, good_name):
    '''
    转换数据，统计每一个值的黑白个数
    :param data:
    :param flag_name:
    :param factor_name:
    :param total_name:
    :param bad_name:
    :param good_name:
    :return:
    '''
    if flag_name != '':
        # 只读取factor_name和flag_name这两个特征的值
        data = data[[factor_name, flag_name]]
        # 确保数据的flag_name是二元化，并且不会有空值。
        data = verify_df_two(data, flag_name, factor_name)
        if len(data) != 0:
            # 根据 flag_name,factor_name聚合，统计flag_name的数量
            data = data[flag_name].groupby([data[factor_name], data[flag_name]]).count()
            # 把series转化成新的 dataframe
            data = data.unstack()
            data = data.reset_index()
            # 定义新的data列名
            data.columns = [factor_name, 'good', 'bad']

            # 将factor_name数据的值类型进行校验，看是不是数值型，然后转化成float.
            data[factor_name] = data[factor_name].apply(lambda x: verify_factor(x))
            # 把data按照factor_name进行升序排序。
            data = data.sort_values(by=[factor_name], ascending=True)
            # 空缺值用0填补
            data = data.fillna(0)
            # 对data新增total字段
            data['total'] = data['good'] + data['bad']
            # 将data的factor_name字段改成str类型
            data[factor_name] = data[factor_name].astype(str)
    else:
        data = map(lambda x: x.upper(), data[factor_name].astype(str))
        verify_df_multiple(data, factor_name, total_name, bad_name, good_name)
        if len(data[factor_name]) != len(set(data[factor_name])):
            data = fun_group_by(data, factor_name, bad_name, good_name)
    print
    'universal_df'
    return data


def Best_KS_Bin(path='', data=pd.DataFrame(), sep=',', flag_name='', factor_name='name', total_name='total',
                bad_name='bad', good_name='good', piece=5, rate=0.05, not_in_list=[]):
    time0 = time.time()
    if len(data) != 0:
        # 如果factor_name是字符串类型，那就全部转化成大写。
        data[factor_name] = map(lambda x: x.upper(), data[factor_name].astype(str))
    elif path != '':
        # 如果path不为空，那么就从path里加载数据
        data = path_df(path, sep, factor_name)
        data[factor_name] = map(lambda x: x.upper(), data[factor_name].astype(str))
    else:
        data = pd.DataFrame()
        print
        'Error: there is no data'
        time1 = time.time()
        print
        'spend time(s):', round(time1 - time0, 0)
        return data

    # 这里就是返回数据里factor_name列数据的每个值的统计
    data = universal_df(data, flag_name, factor_name, total_name, bad_name, good_name)

    # 总的样本数
    total_all = sum(data['total'])
    # 白名单个数
    good_all = sum(data['good'])
    # 黑名单个数
    bad_all = sum(data['bad'])
    if len(data) != 0:
        not_list = map(lambda x: x.upper(), not_in_list)
        if not_in_list:
            not_name = not_list
            if 'NA' in not_list or 'NAN' in not_list or '' in not_list:
                not_name = not_list + ['NAN']
            elif ' ' in not_list:
                not_name = not_list + ['MISSING']
            na_df = data[data[factor_name].isin(not_name)]
            date_df = data.drop(data[data[factor_name].isin(not_name)].index)
            if (0 in na_df[good_name]) or (0 in na_df[bad_name]):
                not_value = list(set(
                    list(na_df[na_df[good_name] == 0][factor_name]) + list(na_df[na_df[bad_name] == 0][factor_name])))
                print
                "Warning: the count of good or bad for the value in 'not_in_list' is 0. The value (" + str(
                    not_value) + ") will not get the separate bin. "
                na_df_new = na_df[na_df[factor_name].isin(not_value)]
                na_df = na_df.drop(na_df[na_df[factor_name].isin(not_value)].index)
                na_df.index = range(len(na_df))
                na_df_new[factor_name] = na_df_new[factor_name].map(lambda x: verify_factor(x))
                date_df[factor_name] = date_df[factor_name].map(lambda x: verify_factor(x))
                date_df = na_df_new.append(date_df)
                date_df = date_df.sort_values(by=factor_name, ascending=True)
                type_len = list(set(map(lambda x: type(x), list(date_df[factor_name]))))
                if len(type_len) > 1:
                    other_df = date_df[date_df[factor_name].apply(lambda x: type(x) == str)]
                    date_df = date_df[date_df[factor_name].apply(lambda x: type(x) == float)]
                    date_df = other_df.append(date_df)
        else:
            # 在not_in_list不为空的时候，执行如下逻辑
            na_df = pd.DataFrame()
            date_df = data
        # 重新定义data_df的index
        date_df.index = range(len(date_df))
        if len(date_df) > 0:
            # 计算分箱
            bin_df = all_information(date_df, na_df, piece, rate, factor_name, total_name, bad_name, good_name,
                                     total_all, good_all, bad_all)
        else:
            time1 = time.time()
            print
            'spend time(s):', round(time1 - time0, 0)
            return data
        time1 = time.time()
        # 统计分箱消耗时长
        print
        'spend time(s):', round(time1 - time0, 0)
        return bin_df
    else:
        time1 = time.time()
        print
        'spend time(s):', round(time1 - time0, 0)
        return data


###############################################对KS分箱之后进行IV排名#########################################
def sort_band_by_iv():
    tmp_df = pd.DataFrame()
    indexvalue = 1
    for filename in os.listdir('/home/chx/eda/band_result'):
        if 'csv' in filename:
            print
            filename
            try:
                band_result = pd.read_csv('/home/chx/eda/band_result/%s' % filename)
                ks = band_result['KS'].max()
                iv_sum = band_result['IV'].sum()
                df = pd.DataFrame({
                    'band': [filename],
                    'ks': [ks],
                    'iv_sum': [iv_sum]
                })
                tmp_df = tmp_df.append(df)
            except Exception as err:
                pass

    tmp_df.reset_index(drop=True, inplace=True)
    tmp_df.info()
    tmp_df.sort_values(by=['iv_sum'], ascending=False, inplace=True)
    print
    tmp_df
    tmp_df.to_csv('/home/chx/eda/IVSort/IV.csv', index=False)


####################################################数据合并#####################################################
# 数据合并
# 就是开房次数和异性同住次数特征表进行合并,并且将数据合并之后的数据保存到本地。
def merge_data(lgzsPath, yxtzPath):
    lgzs_data = pd.read_csv(lgzsPath)
    yxtz_data = pd.read_csv(yxtzPath)
    result_data = pd.merge(yxtz_data, lgzs_data, how='inner', left_on='gmsfhm_rzsj', right_on='gmsfhm_rzsj')
    result_data.rename(columns={'label_x': 'label'}, inplace=True)
    now_time = time.strftime('%Y%m%d', time.localtime(time.time()))
    result_data.to_csv('/home/chx/data/input/new/yxtz_lgzs_merge_%s.csv' % now_time, index=False)


###################################################KS分箱的主类#################################################
class KS_Bin():
    def __init__(self, path, flag, notBandColList):
        '''
        :param path: 数据源路径
        :param flag: 目标值1-0值
        :param colList: 需要分箱的数据列
        '''

        line = os.popen("head -1 %s" % path)
        line = line.readlines()[0]
        if "$" in line:
            self.df = pd.read_csv(path, sep='$', engine='c')
        else:
            self.df = pd.read_csv(path, sep=',', engine='c')
        if 'bad' in self.df['label'].drop_duplicates().values:
            self.df[flag] = self.df[flag].map(lambda x: 1 if x == 'bad' else 0)

        self.flag = flag
        self.path = path
        not_band_list = []
        for col in self.df.columns.tolist():
            if col not in notBandColList:
                not_band_list.append(col)
        self.colList = not_band_list
        print
        self.colList

    def to_band(self):
        for col in tqdm(self.colList):
            ks_data = Best_KS_Bin(data=self.df, flag_name=self.flag, factor_name=col)
            # 将分箱数据导出来

            self.binData_csv(ks_data, '/home/chx/eda/band_result/%s_binResult.csv' % col)
            # 用WOE值代替分类值
            for row in ks_data.index:
                bin = ks_data.loc[row].Bin
                woe = ks_data.loc[row].Woe
                binStart = float(bin.split(',')[0][1:])
                binEnd = float(bin.split(',')[1][:-1])
                self.df[col] = self.df[col].map(lambda x: float(x))
                # 用WOE值代替原来的值
                self.df.loc[(self.df[col] >= binStart) & (self.df[col] <= binEnd), '%s_band' % col] = woe
        print
        'save data'
        self.save_band_data()

    def binData_csv(self, df, csvPath):
        df.to_csv(csvPath, index=False)

    def save_band_data(self):
        '''
          这里就是把分箱之后的字段提取出，作为新的数据进行保存
        '''
        band_list = []
        # 这两个字段现在写死了，看后期怎么玩，其实可以拿出来，当做参数，这样子就可以通用化。
        # 目前只是我们的业务，所以自己写了。
        band_list.append('gmsfhm_rzsj')
        band_list.append('label')
        for col in self.df.columns.tolist():
            if 'band' in col:
                band_list.append(col)

        band_data = self.df[band_list]
        filename = self.path.split('/')[-1]
        filename = filename.split('.')[0] + '_band'
        band_data.to_csv('/home/chx/data/input/new/%s.csv' % filename, index=False)


if __name__ == "__main__":
    data = pd.read_csv('application_test.csv')
    data['FLAG_OWN_CAR'] = data['FLAG_OWN_CAR'].map(lambda x: 1 if x == 'Y' else 0)

    Best_KS_Bin(data=data, factor_name='AMT_INCOME_TOTAL', flag_name='FLAG_OWN_CAR')

    data[['FLAG_OWN_CAR', 'AMT_INCOME_TOTAL']].head()

