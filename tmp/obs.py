#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/5 15:55
# @Author  : AndrewMa
# @File    : obs.py


# Create a dataframe with the variable values replaced by vimp
def var_woe(x, bin_dic, not_in_list):
    val = None
    if str(x) in not_in_list and pd.isnull(x) is False:
        for woe in bin_dic:
            if float(bin_dic[woe][0].lstrip().rstrip()) == x:
                val = woe
    elif pd.isnull(x):
        for woe in bin_dic:
            if bin_dic[woe] == ['nan']:
                val = woe
    else:
        for woe in bin_dic:
            end_points = bin_dic[woe]
            if end_points[0].lstrip().rstrip() not in not_in_list:
                if end_points[0].lstrip().rstrip() == '-inf':
                    if x <= float(end_points[1].lstrip().rstrip()):
                        val = woe
                elif end_points[1].lstrip().rstrip() == 'inf':
                    if x > float(end_points[0].lstrip().rstrip()):
                        val = woe
                elif (x > float(end_points[0].lstrip().rstrip())) & (x <= float(end_points[1].lstrip().rstrip())):
                    val = woe
    return val


# Calculate variable WOE and replace variable value with WOE


def df_woe(flag_name, data=pd.DataFrame(), data1=pd.DataFrame(), bad_name='bad', good_name='good', con_rate=0.90,
           piece=5, rate=0.05, min_bin_size=50, not_in_list=[], not_var_list=[], flag_var_list=[]):
    """
    :param flag_name: bad flag
    :param data: target data
    :param piece: number of bins in final result
    :param rate: number of minimum percent in one bin
    :param not_in_list: special values should be treated
    :parm not_var_list: special variables should be treated
    :return:
    """
    # data_woe = pd.DataFrame()
    data_woe = data[flag_var_list]
    if len(data1) > 0:
        data_woe1 = data1[flag_var_list]

    data_bin = pd.DataFrame()
    if len(data) == 0:
        print('Original input data is empty')
        return pd.DataFrame()
    var_list = data.columns
    not_var_list.extend([flag_name])
    not_var_list.extend(not_in_list)
    not_var_list.append('time_stamp')
    not_in_list.extend(['None', 'nan'])
    not_max_var = []
    for var in data.columns:
        percent = data[var].value_counts(normalize=True, dropna=False)
        # 判定单一值比率
        if percent.max() >= con_rate:
            not_max_var.append(var)
    target_var_list = list(set(var_list) - set(not_var_list) - set(not_max_var))
    #    target_var_list=col
    iv_list = []
    ks_list = []
    target_var_list1 = []

    if len(target_var_list) == 0:
        print('No variable available for analysis')
        return pd.DataFrame()
    iter = 0
    for var in target_var_list:

        print(var)
        try:
            var_stat, not_in_list_1 = Best_KS_Bin(flag_name, var, data, bad_name, good_name, piece, rate, min_bin_size,
                                                  not_in_list)
            if len(var_stat) > 0:
                if len(var_stat['WOE']) != len(set(var_stat['WOE'])):
                    var_stat.ix[var_stat['Bin'] == 'NA', 'WOE'] = var_stat.ix[
                                                                      var_stat['Bin'] == 'NA', 'WOE'] + 0.0000001
                var_stat['var'] = var

                var_stat['WOE'] = var_stat[['total_count', 'WOE']].apply(
                    lambda x: 0 if x[0] < len(data) * 0.05 else x[1], axis=1)
                bin_dic = dict(zip(var_stat['WOE'], var_stat['Bin']))
                for woe in bin_dic:
                    match_case = re.compile("\(|\)|\[|\]")
                    end_points = match_case.sub('', bin_dic[woe]).split(', ')
                    bin_dic[woe] = end_points
                data_woe[var] = list(
                    map(lambda x: var_woe(x, bin_dic, not_in_list_1), data[var].map(lambda x: float(x))))
                if len(data1) > 0:
                    data_woe1[var] = list(
                        map(lambda x: var_woe(x, bin_dic, not_in_list_1), data1[var].map(lambda x: float(x))))
                ivv = list(var_stat['IV'])
                while float('inf') in ivv:
                    ivv.remove(float('inf'))
                iv = sum(ivv)
                ks = max(var_stat['KS'])
                data_bin = pd.concat([data_bin, var_stat])
                # info_dic.update({var: [iv, ks]})
                iv_list.append(iv)
                ks_list.append(ks)
                iter += 1
                print(iter)
                target_var_list1.append(var)
            else:
                #                iv_list.append('nan')
                #                ks_list.append('nan')
                print(var, ' Should be checked')
        except:
            print(var + '--error')
            pass

    data_stat = pd.DataFrame({'var': target_var_list1, 'iv': iv_list, 'ks': ks_list}).sort_values(by='iv',
                                                                                                  ascending=False)
    if len(data1) > 0:
        data_woe.replace(
            ['non', 'none', 'None', 'NONE', 'null', 'NULL', 'Null', '"null"', '[]', '[ ]', '{}', '{ }', ' ', 'nu',
             np.nan], 0, inplace=True)
        data_woe1.replace(
            ['non', 'none', 'None', 'NONE', 'null', 'NULL', 'Null', '"null"', '[]', '[ ]', '{}', '{ }', ' ', 'nu',
             np.nan], 0, inplace=True)
        return data_woe, data_woe1, data_bin, data_stat
    else:
        data_woe.replace(
            ['non', 'none', 'None', 'NONE', 'null', 'NULL', 'Null', '"null"', '[]', '[ ]', '{}', '{ }', ' ', 'nu',
             np.nan], 0, inplace=True)
        return data_woe, data_bin, data_stat


# Best_KS_Bin(flag_name, factor_name, xx, bad_name='bad', good_name='good', piece=5, rate=0.05, min_bin_size=50, not_in_list=[])
#
# df_woe(flag_name, data, bad_name='bad', good_name='good', con_rate = 0.90, piece=5, rate=0.05, min_bin_size=50, not_in_list=[], not_var_list=[], flag_var_list=[])

class f_bin_woe:
    def __init__(self):
        self.flag = ''
        self.model = pd.DataFrame()
        self.col = []
        self.cor_method = 'pearson'
        self.p_cri = 0.5
        self.ivv = pd.DataFrame({'v': [], 'iv': [], 'i_index': [], 'woe_t': []})
        self.id = ''
        self.ce = []
        self.obj = pd.DataFrame({'n': [], 'obj': [], 're': []})
        self.N = 5
        self.coo = []

    def load(self, df, a, b, c, d):
        a = copy.copy(a)
        b = copy.copy(b)
        c = copy.copy(c)
        if b in a:
            a.remove(b)
        if c in a:
            a.remove(c)
        self.model = df[a + [b] + [c]]
        self.flag = b
        self.col = list(set(a))
        self.id = c
        self.model.replace(
            ['non', 'none', 'None', 'NONE', 'null', 'NULL', 'Null', '"null"', '[]', '[ ]', '{}', '{ }', ' ', 'nu'],
            np.nan, inplace=True)
        self.N = d

    # 计算iv最大的woe

    def df_dis(self, df, y):
        df = copy.copy(df)
        for x in self.col:
            if df[x].dtype == 'object':
                df_agg = Cal_WOE(df, x, y)
                ob = pd.DataFrame({'n': [x], 'obj': [list(df_agg.index)], 're': [list(df_agg[x + '_woe'])]})
                self.obj = self.obj.append(ob, ignore_index=True)
                df[x] = df[x].replace(df_agg.index, df_agg[x + '_woe'])
        return df


def dwoe(m, dddf2, col, not_in_list=[]):
    for k in col:
        cd = dddf2[dddf2['var'] == k].copy()
        cd['1'] = range(len(cd))
        cd['WOE'] = cd['WOE'] + cd['1'] / 100000
        dc = dict()
        for l in range(len(cd)):
            dc[list(cd['WOE'])[l]] = cd['Bin'][l].replace(')', '').replace('(', '').replace(']', '').split(',')
            if len(dc[list(cd['WOE'])[l]]) == 1:
                dc[list(cd['WOE'])[l]] = dc[list(cd['WOE'])[l]][0]
        for woe in dc:
            if len(dc[woe]) != 2:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', dc[woe]).split(', ')
                dc[woe] = end_points

        for l in dc:
            if len(dc[l]) == 1:
                not_in_list.append(dc[l][0])

        m[k] = list(map(lambda x: var_woe(x, dc, not_in_list), m[k].map(lambda x: float(x))))
    return m


def dwoe1(m, dddf2, col, not_in_list=[]):
    for k in col:
        cd = dddf2[dddf2['var'] == k].copy()
        cd['1'] = range(len(cd))
        cd['WOE'] = cd['WOE']
        dc = dict()
        for l in range(len(cd)):
            dc[list(cd['WOE'])[l]] = cd['Bin'][l].replace(')', '').replace('(', '').replace(']', '').split(',')
            if len(dc[list(cd['WOE'])[l]]) == 1:
                dc[list(cd['WOE'])[l]] = dc[list(cd['WOE'])[l]][0]
        for woe in dc:
            if len(dc[woe]) != 2:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', dc[woe]).split(', ')
                dc[woe] = end_points

        for l in dc:
            if len(dc[l]) == 1:
                not_in_list.append(dc[l][0])

        m[k] = list(map(lambda x: var_woe(x, dc, not_in_list), m[k].map(lambda x: float(x))))
    return m


# def vvar_woe(x, bin_dic, not_in_list):
#    val = None
#    if str(x) in not_in_list and pd.isnull(x) is False:
#        bin_dicc=dict()
#        for k in not_in_list:
#            for m in bin_dic:
#                if bin_dic[m]==k:
#                    bin_dicc[m]=k
#        for vimp in bin_dicc:
#            if float(bin_dicc[vimp]) == x:
#                val = vimp
#    elif pd.isnull(x):
#        for vimp in bin_dic:s
#            if bin_dic[vimp] == ['nan']:
#                val = vimp
#    else:
#        for vimp in bin_dic:
#            end_points = bin_dic[vimp]
#            if end_points==['nan']:
#                if  pd.isnull(x):
#                    val=vimp
#            elif end_points not in not_in_list:
#                if end_points[0].lstrip().rstrip() == '-inf':
#                    if x <= float(end_points[1]):
#                        val = vimp
#                elif end_points[1].lstrip().rstrip() == 'inf':
#                    if x > float(end_points[0]):
#                        val = vimp
#                elif (x > float(end_points[0])) & (x <= float(end_points[1])):
#                    val = vimp
#    return val
#
#
#

def woe_cal(m, col, y, bad_name, good_name):
    iv = []
    for fac in col:
        na_df = group_by_df(m, y, fac, bad_name, good_name, False)
        total_good = sum(na_df[good_name])
        total_bad = sum(na_df[bad_name])
        na_good_percent = na_df[good_name] / float(total_good)
        na_bad_percent = na_df[bad_name] / float(total_bad)
        na_indicator = pd.DataFrame({'Bin': list(na_df.ix[:, 0]), 'KS': [None] * len(na_df),
                                     'WOE': list(np.log(na_bad_percent / na_good_percent)),
                                     'IV': list(
                                         (na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                                     'total_count': list(na_df[good_name] + na_df[bad_name]),
                                     'bad_rate': list(na_df[bad_name] / (na_df[good_name] + na_df[bad_name]))})
        na_indicator['var'] = fac
        iv.append(na_indicator['IV'].sum())
    return pd.DataFrame({'var': col, 'iv': iv})