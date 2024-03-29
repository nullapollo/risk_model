import numpy as np
import pandas as pd
import re
import math
import time
import itertools
from itertools import combinations
from numpy import array
from math import sqrt
from multiprocessing import Pool

def get_max_ks(date_df, start, end, rate, bad_name, good_name):
    ks = ''
    if end == start:
        return ks
    bad = date_df.loc[start:end, bad_name]
    good = date_df.loc[start:end, good_name]
    bad_good_cum = list(abs(np.cumsum(bad / sum(bad)) - np.cumsum(good / sum(good))))
    if bad_good_cum:
        ks = start + bad_good_cum.index(max(bad_good_cum))

    return ks


def cut_while_fun(start, end, piece, date_df, rate, bad_name, good_name, counts):
    '''
    :param start: 起始位置0
    :param end: 终止位置
    :param piece: 起始切分组
    :param date_df: 数据集
    :param rate: 最小分组占比
    :param bad_name: Y标签1对应列名
    :param good_name: Y标签0对应列名
    :param counts: 默认从1计数
    :return:
    '''
    point_all = []
    if counts >= piece or len(point_all) >= pow(2, piece - 1):
        return []
    ks_point = get_max_ks(date_df, start, end, rate, bad_name, good_name)
    if ks_point:
        if ks_point != '':
            t_up = cut_while_fun(start, ks_point, piece, date_df, rate, bad_name, good_name, counts + 1)
        else:
            t_up = []
        t_down = cut_while_fun(ks_point + 1, end, piece, date_df, rate, bad_name, good_name, counts + 1)
    else:
        t_up = []
        t_down = []
    point_all = t_up + [ks_point] + t_down
    return point_all


def ks_auto(date_df, piece, rate, bad_name, good_name):
    t_list = list(set(cut_while_fun(0, len(date_df) - 1, piece - 1, date_df, rate, bad_name, good_name, 1)))
    # py2
    # ks_point_all = [0] + filter(lambda x: x != '', t_list) + [len(date_df) - 1]
    # py3
    ks_point_all = [0] + list(filter(lambda x: x != '', t_list)) + [len(date_df) - 1]
    return ks_point_all


def get_combine(t_list, date_df, piece):
    t1 = 0
    t2 = len(date_df) - 1
    list0 = t_list[1:len(t_list) - 1]
    combine = []
    if len(t_list) - 2 < piece:
        c = len(t_list) - 2
    else:
        c = piece - 1
    list1 = list(itertools.combinations(list0, c))
    if list1:
        combine = map(lambda x: sorted(x + (t1 - 1, t2)), list1)

    return combine


def cal_iv(date_df, items, bad_name, good_name, rate, total_all, mono=True):
    iv0 = 0
    total_rate = [sum(date_df.ix[x[0]:x[1], bad_name] + date_df.ix[x[0]:x[1], good_name]) * 1.0 / total_all for x in
                  items]
    if [k for k in total_rate if k < rate]:
        return 0
    bad0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items)))
    good0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items)))
    bad_rate0 = bad0 * 1.0 / (bad0 + good0)
    if 0 in bad0 or 0 in good0:
        return 0
    good_per0 = good0 * 1.0 / sum(date_df[good_name])
    bad_per0 = bad0 * 1.0 / sum(date_df[bad_name])
    woe0 = list(map(lambda x: math.log(x, math.e), good_per0 / bad_per0))
    if mono:
        if sorted(woe0, reverse=False) == list(woe0) and sorted(bad_rate0, reverse=True) == list(bad_rate0):
            iv0 = sum(woe0 * (good_per0 - bad_per0))
        elif sorted(woe0, reverse=True) == list(woe0) and sorted(bad_rate0, reverse=False) == list(bad_rate0):
            iv0 = sum(woe0 * (good_per0 - bad_per0))
    else:
        iv0 = sum(woe0 * (good_per0 - bad_per0))
    return iv0


def choose_best_combine(date_df, combine, bad_name, good_name, rate, total_all, mono=True):
    z = [0] * len(combine)
    for i in range(len(combine)):
        item = combine[i]
        z[i] = list(zip(map(lambda x: x + 1, item[0:len(item) - 1]), item[1:]))
    iv_list = list(map(lambda x: cal_iv(date_df, x, bad_name, good_name, rate, total_all, mono=mono), z))
    iv_max = max(iv_list)
    if iv_max == 0:
        return ''
    index_max = iv_list.index(iv_max)
    combine_max = z[index_max]
    return combine_max


def verify_woe(x):
    if re.match('^\d*\.?\d+$', str(x)):
        return x
    else:
        return 0


def best_df(date_df, items, na_df, factor_name, bad_name, good_name, total_all, good_all, bad_all, var_val_type):
    df0 = pd.DataFrame()
    if items:
        if var_val_type == 'number':
            piece0 = list(map(
                lambda x: '(' + str(date_df.ix[x[0], factor_name]) + ',' + str(date_df.ix[x[1], factor_name]) + ')',
                items))
        else:
            piece0 = list(map(
                lambda x: '(' + ','.join(date_df.ix[x[0]:x[1], factor_name]) + ')',
                items))
        bad0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items))
        good0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items))
        if len(na_df) > 0:
            piece0 = array(
                list(piece0) + list(map(lambda x: '(' + str(x) + ',' + str(x) + ')', list(na_df[factor_name]))))
            bad0 = array(list(bad0) + list(na_df[bad_name]))
            good0 = array(list(good0) + list(na_df[good_name]))
        else:
            piece0 = array(list(piece0))
            bad0 = array(list(bad0))
            good0 = array(list(good0))
        total0 = bad0 + good0
        total_per0 = total0 * 1.0 / total_all
        bad_rate0 = bad0 * 1.0 / total0
        good_rate0 = 1 - bad_rate0
        good_per0 = good0 * 1.0 / good_all
        bad_per0 = bad0 * 1.0 / bad_all
        df0 = pd.DataFrame(
            list(zip(piece0, total0, bad0, good0, total_per0, bad_rate0, good_rate0, good_per0, bad_per0)),
            columns=[factor_name, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate',
                     'Good_Pcnt', 'Bad_Pcnt'])
        df0 = df0.sort_values(by='Bad_Rate', ascending=False)
        df0.index = range(len(df0))
        bad_per0 = array(list(df0['Bad_Pcnt']))
        good_per0 = array(list(df0['Good_Pcnt']))
        bad_rate0 = array(list(df0['Bad_Rate']))
        good_rate0 = array(list(df0['Good_Rate']))
        bad_cum = np.cumsum(bad_per0)
        good_cum = np.cumsum(good_per0)
        woe0 = list(map(lambda x: math.log(x, math.e), good_per0 / bad_per0))
        if 'inf' in str(woe0):
            woe0 = list(map(lambda x: verify_woe(x), woe0))
        iv0 = woe0 * (good_per0 - bad_per0)
        gini = 1 - pow(good_rate0, 2) - pow(bad_rate0, 2)
        df0['Bad_Cum'] = bad_cum
        df0['Good_Cum'] = good_cum
        df0["Woe"] = woe0
        df0["IV"] = iv0
        df0['Gini'] = gini
        df0['KS'] = abs(df0['Good_Cum'] - df0['Bad_Cum'])

    return df0


def all_information(date_df, na_df, piece, rate, factor_name, bad_name, good_name, total_all, good_all, bad_all,
                    var_val_type, mono=True):
    '''
        :param date_df:非异常值数据
        :param na_df 空值数据
        :param piece:切割组数
        :param rate:最小分组比例
        :param factor_name:变量名
        :param bad_name:坏的列名
        :param good_name:好的列名
        :param total_all:总样本数
        :param good_all:好的总样本数
        :param bad_all:坏的总样本数
        :return:
    '''
    p_sort = range(piece + 1)
    p_sort = [i for i in p_sort]
    p_sort.sort(reverse=True)
    t_list = ks_auto(date_df, piece, rate, bad_name, good_name)
    if not t_list:
        df1 = pd.DataFrame()
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1
    df1 = pd.DataFrame()
    for c in p_sort[:piece - 1]:
        combine = list(get_combine(t_list, date_df, c))
        best_combine = choose_best_combine(date_df, combine, bad_name, good_name, rate, total_all, mono=mono)
        df1 = best_df(date_df, best_combine, na_df, factor_name, bad_name, good_name, total_all, good_all, bad_all,
                      var_val_type)
        if len(df1) != 0:
            gini = sum(df1['Gini'] * df1['Total_Num'] / sum(df1['Total_Num']))
            print('piece_count:', str(len(df1)))
            print('IV_All_Max:', str(sum(df1['IV'])))
            print('Best_KS:', str(max(df1['KS'])))
            print('Gini_index:', str(gini))
            print(df1)
            return df1
    if len(df1) == 0:
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1


def verify_factor(x):
    if x in ['NA', 'NAN', '', ' ', 'MISSING', 'NONE', 'NULL']:
        return 'NAN'
    if re.match('^\-?\d*\.?\d+$', x):
        x = float(x)

    return x


def path_df(path, sep, factor_name):
    data = pd.read_csv(path, sep=sep)
    data[factor_name] = data[factor_name].astype(str).map(lambda x: x.upper())
    data[factor_name] = data[factor_name].apply(lambda x: re.sub(' ', 'MISSING', x))

    return data


def verify_df_multiple(date_df, factor_name, total_name, bad_name, good_name):
    """
    :param date_df: factor_name,....
    :return: factor_name,good_name,bad_name
    """
    date_df = date_df.fillna(0)
    cols = date_df.columns
    if total_name in cols:
        date_df = date_df[date_df[total_name] != 0]
        if bad_name in cols and good_name in cols:
            date_df_check = date_df[date_df[good_name] + date_df[bad_name] - date_df[total_name] != 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is not equal to the sum of bad & good amounts')
                print(date_df_check)
                return date_df
        elif bad_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[bad_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is smaller than bad amounts')
                print(date_df_check)
                return date_df
            date_df[good_name] = date_df[total_name] - date_df[bad_name]
        elif good_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[good_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is smaller than good amounts')
                print(date_df_check)
                return date_df
            date_df[bad_name] = date_df[total_name] - date_df[good_name]
        else:
            print('Error: lack of bad or good data')
            date_df = pd.DataFrame()
            return date_df
        del date_df[total_name]
    elif bad_name not in cols:
        print('Error: lack of bad data')
        date_df = pd.DataFrame()
        return date_df
    elif good_name not in cols:
        print('Error: lack of good data')
        date_df = pd.DataFrame()
        return date_df
    date_df[good_name] = date_df[good_name].astype(int)
    date_df[bad_name] = date_df[bad_name].astype(int)
    date_df = date_df[date_df[bad_name] + date_df[good_name] != 0]
    date_df[factor_name] = date_df[factor_name].map(verify_factor)
    date_df = date_df.sort_values(by=[factor_name], ascending=True)
    date_df[factor_name] = date_df[factor_name].astype(str)
    if len(date_df[factor_name]) != len(set(date_df[factor_name])):
        df_bad = date_df.groupby(factor_name)[bad_name].agg([(bad_name, 'sum')]).reset_index()
        df_good = date_df.groupby(factor_name)[good_name].agg([(good_name, 'sum')]).reset_index()
        good_dict = dict(zip(df_good[factor_name], df_good[good_name]))
        df_bad[good_name] = df_bad[factor_name].map(good_dict)
        df_bad.index = range(len(df_bad))
        date_df = df_bad

    return date_df


def verify_df_two(data_df, flag_name, factor_name, good_name, bad_name):
    """
    :param data_df, factor_name, flag_name
    :return: factor_name,good_name,bad_name
    预处理：将df_按照flag(0-1)groupby，并按照var_value排序
    """

    data_df = data_df[-pd.isnull(data_df[flag_name])]
    if len(data_df) == 0:
        print('Error: the data is wrong')
        return data_df
    check = data_df[data_df[flag_name] > 1]
    if len(check) != 0:
        print('Error: there exits the number bigger than one in the data')
        data_df = pd.DataFrame()
        return data_df
    if flag_name != '':
        try:
            data_df[flag_name] = data_df[flag_name].astype(int)
        except:
            print('Error: the data is wrong')
            data_df = pd.DataFrame()
            return data_df
    data_df = data_df[flag_name].groupby(
        [data_df[factor_name], data_df[flag_name]]).count().unstack().reset_index().fillna(0)
    data_df.columns = [factor_name, good_name, bad_name]
    data_df[factor_name] = data_df[factor_name].apply(verify_factor)
    # data_df_1=data_df[data_df[factor_name].apply(lambda x: str(x).find('NAN')>=0 or str(x).find('E')>=0)]
    # data_df_2=data_df[~data_df[factor_name].apply(lambda x: str(x).find('NAN')>=0 or str(x).find('E')>=0)]
    # data_df_2 = data_df_2.sort_values(by=[factor_name], ascending=True)
    # data_df=pd.concat([data_df_1,data_df_2])
    data_df.index = range(len(data_df))
    data_df[factor_name] = data_df[factor_name].astype(str)
    return data_df


def universal_df(data, flag_name, factor_name, total_name, bad_name, good_name):
    if flag_name != '':
        data = data[[factor_name, flag_name]]
        data = verify_df_two(data, flag_name, factor_name, good_name, bad_name)
    else:
        data = verify_df_multiple(data, factor_name, total_name, bad_name, good_name)
    return data


def Best_KS_Bin(path='', data=pd.DataFrame(), sep=',', flag_name='', factor_name='name', total_name='total',
                bad_name='bad', good_name='good', bin_num=5, rate=0.05, not_in_list=[], value_type=True,
                var_type='number', mono=True):
    """
    :param flag_name:Y标签
    :param factor_name: 变量名
    :param total_name:
    :param bad_name: bad
    :param good_name: good
    :param bin_num:切割组数,默认5
    :param rate:分组占比不得小于
    :param not_in_list:['NaN', '-1.0', '', '-1']
    :param value_type: True is numerical; False is nominal
    """
    # none_list = ['NA', 'NAN', '', ' ', 'MISSING', 'NONE', 'NULL']
    if path != '':
        # 若直接调用分箱则输入数据路径
        data = path_df(path, sep, factor_name)
    elif len(data) == 0:
        print('Error: there is no data')
        return data
    data = data.copy()
    data[factor_name] = data[factor_name].apply(lambda x: str(x).upper().replace(',', '_'))
    data = universal_df(data, flag_name, factor_name, total_name, bad_name, good_name)
    if len(data) == 0:
        return data
    good_all = sum(data[good_name])
    bad_all = sum(data[bad_name])
    total_all = good_all + bad_all
    if not_in_list:
        # 空值分组
        not_name = [str(k).upper() for k in not_in_list]
        # for n0 in none_list:
        #     print (n0)
        #     if n0 in not_name:
        #         not_name += ['NAN']  # todo
        #         print(not_name)
        #         break
        na_df = data[data[factor_name].isin(not_name)]
        if (0 in na_df[good_name]) or (0 in na_df[bad_name]):
            not_value = list(
                set(list(na_df[na_df[good_name] == 0][factor_name]) + list(na_df[na_df[bad_name] == 0][factor_name])))
            na_df = na_df.drop(na_df[na_df[factor_name].isin(not_value)].index)
            na_df.index = range(len(na_df))
        not_list = list(set(na_df[factor_name]))
        date_df = data[-data[factor_name].isin(not_list)]
    else:
        na_df = pd.DataFrame()
        date_df = data
    if len(date_df) == 0:
        print('Error: the data is wrong.')
        data = pd.DataFrame()
        return data
    if value_type:
        date_df = date_df.copy()
        if var_type != 'str':
            date_df[factor_name] = date_df[factor_name].apply(verify_factor)
        type_len = set([type(k) for k in list(date_df[factor_name])])
        if len(type_len) > 1:
            str_df = date_df[date_df[factor_name].map(lambda x: type(x) == str)]
            number_df = date_df[date_df[factor_name].map(lambda x: type(x) == float)]
            number_df = number_df.sort_values(by=factor_name)
            str_df = str_df.sort_values(by=factor_name)
            date_df = str_df.append(number_df)
        else:
            date_df = date_df.sort_values(by=factor_name)
    else:
        date_df['bad_rate'] = date_df[bad_name] * 1.0 / (date_df[good_name] + date_df[bad_name])
        date_df = date_df.sort_values(by=['bad_rate', factor_name], ascending=False)
    date_df[factor_name] = date_df[factor_name].astype(str)  # todo
    date_df.index = range(len(date_df))
    # date_df.to_csv(factor_name+'.csv')
    # print(date_df)
    # ks分箱
    bin_df = all_information(date_df, na_df, bin_num, rate, factor_name, bad_name, good_name, total_all, good_all,
                             bad_all, var_type, mono=mono)
    return bin_df


def gb_add_woe(data):
    df_gp = data.copy()
    df_gp['pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp = df_gp.sort_values(by=['pct_default'], ascending=False)
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    df_gp['bad_pct'] = df_gp['bad'] / bad_sum
    df_gp['good_pct'] = df_gp['good'] / good_sum
    df_gp['odds'] = df_gp['good_pct'] - df_gp['bad_pct']
    df_gp['Woe'] = np.log(df_gp['good_pct'] / df_gp['bad_pct'])
    df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    df_gp['bad_cnt'] = df_gp['bad'].cumsum()
    df_gp['good_cnt'] = df_gp['good'].cumsum()
    df_gp['b_c_p'] = df_gp['bad_cnt'] / bad_sum
    df_gp['g_c_p'] = df_gp['good_cnt'] / good_sum
    df_gp['KS'] = df_gp['g_c_p'] - df_gp['b_c_p']
    ks_max = df_gp['KS'].max()
    return ks_max, df_gp


def set_tag(x):
    if x == 1:
        y = 1
        z = 0
    else:
        y = 0
        z = 1
    return y, z


def cut_method(data1, factor_name, flag_name, method, n):
    data = data1.copy()
    if method == 'qcut':
        data[factor_name] = pd.qcut(data[factor_name], n)
    elif method == 'cut':
        data[factor_name] = pd.cut(data[factor_name], n)
    elif method == 'uncut':
        data[factor_name] = data[factor_name][:]
    elif method == 'cumsum':
        values = pd.DataFrame(data[factor_name].value_counts()).sort_index()
        values['cum'] = values[factor_name].cumsum()
        # sort_values=np.sort(data[factor_name].value_counts().index)
        sum = data[factor_name].shape[0]
        sd_bin = sum / n
        botton = float(list(values.index)[0]) - 0.001
        values_c = values.copy()
        bin = []
        for i in range(n):
            values_i = values_c[values_c['cum'] <= sd_bin]
            if values_i.shape[0] == 0:
                top = list(values_c.index)[0]
            else:
                top = list(values_i.index)[-1]
            bin.append([botton, top])
            botton = top
            values_c = values_c[values_c.index > top]

            values_c['cum'] = values_c[factor_name].cumsum()
            if values_c.shape[0] == 0:
                break

        bin.append([botton, list(values.index)[-1]])
        data[factor_name] = data[factor_name].map(lambda x: find_bin(x, bin))
    return data


def find_bin(x, list_bin):
    for i in list_bin:
        if x > i[0] and x <= i[1]:
            y = str(i)

    try:
        return y
    except:
        print(x, list_bin)


def loop(df, factor_name, flag_name, method):
    '''
    用于寻找保证单调性下的最大分Bin组数
    :param df:
    :param factor_name:
    :param ex_value:
    :return:
    '''
    find_n = []
    data_for_cut = df.copy()
    for n in range(2, 10, 1):
        print('loop to ', n)
        # try:
        data_1 = cut_method(data_for_cut, factor_name, flag_name, method, n)
        data_gp = data_1.groupby(factor_name).sum()
        data_gp['sort'] = data_gp.index.map(lambda x: float(x.split(',')[0][1:]))
        df_jugde = data_gp.sort_values('sort').drop('sort', axis=1)
        # pct_list=gb_add_woe(df_jugde)[1]['pct_default'].tolist()
        pct_list = list(df_jugde['bad'] / (df_jugde['bad'] + df_jugde['good']))
        if pd.Series(pct_list).is_monotonic_decreasing or pd.Series(pct_list).is_monotonic_increasing:
            find_n.append(n)
        else:
            break
    max_n = max(find_n)
    return max_n


def cut_cumsum_bin(data, factor_name, flag_name, not_in_list, method, mono=True):
    from numpy import array
    data = data[[factor_name, flag_name]].fillna('NAN')
    data['bad'] = data[flag_name].map(lambda x: set_tag(x)[0])
    data['good'] = data[flag_name].map(lambda x: set_tag(x)[1])
    df_ex1 = data[data[factor_name].map(lambda x: str(x) in not_in_list)]  # 分割需要单独切出的bin值(-1，nan),单独值的bin使用[a,a]来表示
    df_ex = df_ex1.copy()
    df_ex[factor_name] = df_ex[factor_name].map(lambda x: '(' + str(x) + ',' + str(x) + ')')
    df_rm1 = data[data[factor_name].map(lambda x: str(x) not in not_in_list)]
    ##
    if mono:
        n = loop(df_rm1, factor_name, flag_name, method)
    else:
        n = max(20, len(set(data[factor_name])))
    # print 'cut',n
    df_rm = df_rm1.copy()
    df_rm[factor_name] = cut_method(df_rm, factor_name, flag_name, method, n)

    # print df_rm
    df = pd.concat([df_ex, df_rm], axis=0)
    df_ = df.groupby(factor_name).sum()[['bad', 'good']]
    df_gp = gb_add_woe(df_)[1]
    df_gp[factor_name] = df_gp.index
    # df_gp['sort'] = df_gp[factor_name].map(lambda x: float(str(x).split(',')[0][1:]))
    df_gp = df_gp.sort_values('pct_default')
    df_new = df_gp.copy()
    df_new['Total_Num'] = df_new['bad'] + df_new['good']
    df_new['Bad_Num'] = df_new['bad']
    df_new['Good_Num'] = df_new['good']
    df_new['Total_Pcnt'] = df_new['Total_Num'] / df_new['Total_Num'].sum()
    df_new['Bad_Rate'] = df_new['Bad_Num'] / df_new['Total_Num']
    df_new['Good_Rate'] = df_new['Good_Num'] / df_new['Total_Num']
    df_new['Good_Pcnt'] = df_new['Good_Num'] / df_new['Good_Num'].sum()
    df_new['Bad_Pcnt'] = df_new['Bad_Num'] / df_new['Bad_Num'].sum()
    df_new['Bad_Cum'] = df_new['b_c_p']
    df_new['Good_Cum'] = df_new['g_c_p']
    df_new['Gini'] = 1 - pow(array(list(df_new['Good_Rate'])), 2) - pow(array(list(df_new['Bad_Rate'])), 2)
    df_new = df_new[[factor_name,
                     'Total_Num',
                     'Bad_Num',
                     'Good_Num',
                     'Total_Pcnt',
                     'Bad_Rate',
                     'Good_Rate',
                     'Good_Pcnt',
                     'Bad_Pcnt',
                     'Bad_Cum',
                     'Good_Cum',
                     'Woe',
                     'IV',
                     'Gini',
                     'KS']]
    df_new[factor_name] = df_new[factor_name].apply(
        lambda x: x if str(x).find('NAN') >= 0 else '(' + str(x) + ',' + str(x) + ')')
    df_new = df_new.sort_values('Bad_Rate', ascending=False)
    df_new = df_new.reset_index(drop=True)
    return df_new
