#!~/anaconda/bin/ python
# -*- coding: utf-8 -*-
# @Time    : 2018/10/22 14:23
# @Author  : Andrewma
# @Site    : 
# @File    : preprocess_function.py
# @Software: PyCharm

# 建模变量转换与处理函数：pandas 针对缺失数据按照浮点型处理（未指定数据格式类型时）

import re
import time
import datetime
import numpy as np
from dateutil.relativedelta import relativedelta


def CareerYear(x):
    # 对工作年限进行转换
    if type(x).__name__ == 'float':
        return -1
    elif x.find("10+") > -1:  # 将"10＋years"转换成 11
        return 11
    elif x.find("< 1") > -1:  # 将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))  # 其余数据，去掉"years"并转换成整数


def DescExisting(x):
    # 将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x, format):
    """
    time.strptime(string, format) 把时间格式字符串，转换为时间元组（每个item都有值的那种）

    :param x:
    :param format:
    :return:
    """
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1', '%Y-%m')))
    else:
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime(x, format)))


def ConvertDateStr2(x, format):
    if str(x) == 'nan':
        return '2099-01-01'
    else:
        return datetime.datetime.strptime(x, format).strftime('%Y-%m-%d')


def MonthGap(earlyDate, lateDate):
    ld = datetime.datetime.strptime(lateDate, '%Y-%m-%d')
    ed = datetime.datetime.strptime(earlyDate, '%Y-%m-%d')
    if ld > ed:
        gap = relativedelta(ld, ed)
        yr = gap.years
        mth = gap.months
        return yr * 12 + mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1  # 缺失值当特殊类别处理
    else:
        return x
