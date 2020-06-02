# encoding:utf-8
import re
from itertools import combinations
import pandas as pd
import numpy as np
import datetime
# import pickle
# from datetime import datetime
# import time
# from collections import Counter
# import operator
# from array import *
# from multiprocessing import Process as ps

#					COPYRIGHT: Risk Management Dept.
#							AUTH: jian.wu
#							2017-12-29
"""
本程序主要用于数据分bin。思想是基于最大KS对变量分裂，然后在满足单调性以及最小bin大小的要求下
基于最大IV对分裂后区间进行合并。
"""

# --
# 更新log
# --
"""
20160827: 更新分裂步, 在分裂步传入参数,保证分裂出来小区间大小被rate以及数据量大小控制
20160830: 更新df_woe, 在变量分bin失败后,程序不会直接跳出
20160830: 更新important_indicator_calculator, 解决一个会导致na的IV为空的bug
20160906: 更新var_woe, 更正里面一个会导致woe化不正确的错误
20160918: 更新var_woe,更正数据为missing时,不能woe化的问题
20160918: 更新df_woe,更正数据为字符串时,导致错误赋值的问题
20161113: 发现当变量集中程度过高时,分bin会失败，导致数据进行woe化将产生空值的bug
20161114: 发现当同一个变量分Bin后存在相同的woe值时,该变量woe化的结果将产生空值。原因在df_woe函数,bin_dic字典的键不能重复
20170118 - v1.1: 1）：更新df_woe的输出，变量指标参数输出从字典形式变成数据框形式；2）：增加对每箱数据样本个数的最小绝对数值的控制，默认为50个
20170118 - v1.2: 对best_ks_knots_helper函数进行了修改。之前存在bug，对数据进行分裂的时候，每一步分裂时候会自动跳过前一步分裂节点之前的节点。如数据有有5个不同值
          0，1，2，3，4.第一步分裂节点为3，那么第二步对0，1，2，3进行分裂时候会跳过2这个节点，2与3肯定会在一个bin中
20170728: 修改因为woe相同导致woe为空的bug
20171212: 添加了urteil函数，用于选出单峰或者u型的分箱

20171212: 添加了df_woe函数，用于输出结果
20171212: 添加了dwoe函数，用于给其他数据集按照已经分好的分箱规则分箱

20180815: 限制分箱的最小箱数，小于等于3箱除非IV很大，不然不予采用
20180815：单峰和单调的比较，如果单峰比单调好百分之三十，则保留单峰，其他情况保留单调

"""




#
#def group_by_df(data, flag_name, factor_name, bad_name, good_name, na_trigger):
#    if len(data) == 0:
#        return pd.DataFrame()
#    data = data[flag_name].groupby([data[factor_name], data[flag_name]]).count()
#    data = data.unstack()
#    data = data.reset_index()
#    data = data.fillna(0)    
#    name=list(data.columns)
#    name=[good_name if x==0 else x for x in name]
#    name=[bad_name if x==1 else x for x in name]
#    data.columns = name
#    if not na_trigger:
#        data[factor_name] = data[factor_name].astype(float)
#    data = data.sort_values(by=[factor_name], ascending=True)
#    data[factor_name] = data[factor_name].astype(str)
#    data = data.reset_index(drop=True)
#
#    pda=pd.DataFrame(columns= [factor_name, good_name, bad_name],index=data.index) 
#
#    for l in [factor_name, good_name, bad_name]:
#        try:
#            pda[l]=data[l]
#        except:
#            pda[l]=0
#    return pda

def group_by_df(data, flag_name, factor_name, bad_name, good_name, na_trigger):
    if len(data) == 0:
        return pd.DataFrame()
    data = data[flag_name].groupby([data[factor_name], data[flag_name]]).count()
    data = data.unstack()
    data = data.reset_index()
    data = data.fillna(0)
    if len(data.columns) == 3:
        data.columns = [factor_name, good_name, bad_name]
        if not na_trigger:
            data[factor_name] = data[factor_name].astype(float)
        data = data.sort_values(by=[factor_name], ascending=True)
        data[factor_name] = data[factor_name].astype(str)
        data = data.reset_index(drop=True)
        return data
    else:
        return pd.DataFrame()

   
    
def verify_df_two(date_df, flag_name):
    
    
    date_df = date_df.dropna(subset=[flag_name])
    
    check = date_df[date_df[flag_name] > 1]
    if len(check) != 0:
        print ('Error: there exits the number bigger than one in the data')
        date_df = pd.DataFrame()
        return date_df
    elif len(date_df) != 0:
        date_df[flag_name] = date_df[flag_name].astype(int)
        return date_df
    else:
        print ('Error: the data is wrong')
        date_df = pd.DataFrame()
        return date_df


def getAllindex(tar_list, item):
    return list(filter(lambda a: tar_list[a] == item, range(0, len(tar_list))))



# the split function part
# since there are the restriction to the position of the best ks ,thus we need to add some transformation to the start
# and end knot in the current function
def best_KS_knot_calculator(data, total_len, good_name, bad_name, start_knot, end_knot, rate):
    # this part may seems to be redundant, you can replace it by passing the total_len in the input part of the function
    # total_len = sum(data[good_name]) + sum(data[bad_name])
    temp_df = data.loc[start_knot:end_knot]
    # the temp len here represents the length of the unique values in the raw data, not only the length of the temp_df
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])
    # since we'd want to make sure the number of elements in each tiny bin should always greater than 5% of the raw
    # data's length, we need to add some restrictions to the start_knot and the end_knot
    start_add_num = sum(
        np.cumsum(temp_df[good_name] + temp_df[bad_name]) < rate * total_len)
    end_add_num = sum(
        np.cumsum(temp_df[good_name] + temp_df[bad_name]) <= temp_len - rate * total_len)
    processed_start_knot = start_knot + start_add_num
    processed_end_knot = start_knot + end_add_num - 1
    if processed_end_knot >= processed_start_knot:
        if sum(temp_df[bad_name]) != 0 and sum(temp_df[good_name]) != 0:
            default_CDF = np.cumsum(temp_df[bad_name]) / sum(temp_df[bad_name])
            undefault_CDF = np.cumsum(temp_df[good_name]) / sum(temp_df[good_name])
            ks_value = max(abs(default_CDF - undefault_CDF).loc[processed_start_knot:processed_end_knot])
            # the index find here is not the final result, we should find the data's position in the outer data set
            index = getAllindex(list(abs(default_CDF - undefault_CDF)), ks_value)
            return temp_df.index[max(index)]
        else:
            return None
    else:
        return None

def best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot, end_knot, current_time):
    # define the base case
    # here, we should first find out the total length of the raw data. Since the elements in the input data
    # represent the count of unique item in the raw data, thus we need to do some transformation to it to find
    # out the length that we need
    # total_len = sum(data[good_name]) + sum(data[bad_name])
    temp_df = data.loc[start_knot:end_knot]
    temp_len = sum(temp_df[good_name]) + sum(temp_df[bad_name])
    # due to the restriction to the number of elements in the tiny bin
    if temp_len < rate * total_len * 2 or current_time >= max_times:
        return []
    new_knot = best_KS_knot_calculator(data, total_len, good_name, bad_name, start_knot, end_knot, rate)
    if new_knot is not None:
        # upper_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot,
        #                                     new_knot - 1, current_time + 1)
        upper_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, start_knot,
                                            new_knot, current_time + 1)
        lower_result = best_ks_knots_helper(data, total_len, max_times, good_name, bad_name, rate, new_knot + 1,
                                            end_knot, current_time + 1)
    else:
        upper_result = []
        lower_result = []
    return upper_result + [new_knot] + lower_result

def new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name):
    # call the helper function to finish the following recursion part
    temp_result_list = best_ks_knots_helper(data_df, total_rec, piece, good_name, bad_name, rate, 0, len(data_df), 0)
    # since we are gonna to use reconstruct the whole function, thus I choose not to add the two marginal points to the
    # result list
    # temp_result_list = temp_result_list + [0, len(data_df)-1]
    aaa=list(filter(lambda x: x is not None, temp_result_list))
    aaa.sort()
    return aaa
def urteil(li):
    if len(li)<4:
        return 1
    else:
        lii=[li[i]-li[i-1] for i in range(len(li))[1:]]
        lii=list(map(lambda x:x if x!=0 else 1, lii))
#        print lii
        zz=np.sign([lii[i]/lii[i-1] for i in range(len(lii))[1:]]).sum()
        if zz in [len(li)-2,len(li)-4]:
            return 1
        else:
            return 0

# the merge function part

# first, we need to define a IV calculator function. The input should be the knots list and the data with unique value's
# black flag count. The result returned by the function will be the IV list of these tiny bins
def IV_calculator(data_df, good_name, bad_name, knots_list,ur):
    # to improve the efficiency of the calculation, I first split the df into a bunch of smaller data frames and put
    # them into a list. Then I Use the map function to do some transformation to calculate the IV value for each small bin
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
    woe_list = list(np.log(good_percent_series / bad_percent_series))
    # here, since we want to make sure the vimp of the result bins is monotonic, thus we add a justification statement
    # here, if it is not monotonic, then it will be discarded and the return will be None
#    if sorted(woe_list) != woe_list and sorted(woe_list, reverse=True) != woe_list:
#        return None
    IV_series = (good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series)
    if np.inf in list(IV_series) or -np.inf in list(IV_series):
        return None
    if ur:
        if urteil(woe_list)==0:
            return None    
        else:
            return sum(IV_series)    

    if sorted(woe_list)==woe_list or sorted(woe_list,reverse=True)==woe_list:
        return sum(IV_series)
    else:
        return None


# combination_helper function
def combine_helper(data_df, good_name, bad_name, piece_num, cut_off_list,ur):
    knots_list = list(combinations(cut_off_list, piece_num - 1))
    # here we do some transformation to the knots list, add the start knot and end knot to all the elements in the list
    # knots_list = map(lambda x: sorted(tuple(set(x + (0, len(data_df) - 1)))), knots_list)
    knots_list = list( map(lambda x: sorted(x + (0, len(data_df) - 1)), knots_list))
    print (len(knots_list))
    IV_for_bins = list(map(lambda x: IV_calculator(data_df, good_name, bad_name, x,ur), knots_list))
    filtered_IV = list(filter(lambda x: x is not None, IV_for_bins))
    if len(filtered_IV) == 0:
        print('There are no suitable division for the data set with ' + str(piece_num) + ' pieces')
        return None,None
    else:
        if len(getAllindex(IV_for_bins, max(filtered_IV))) > 0:
            target_index = getAllindex(IV_for_bins, max(filtered_IV))[0]
            return knots_list[target_index],IV_for_bins[target_index]
        else:
            return None,None


# the cut_off_list here should not contain the start knot 0 and end knot len(data_df)-1, since these knots will be added
# in the later process in function's map part
def combine_tiny_bins0(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    return_piece_num = min(max_piece_num, len(cut_off_list) + 1)
    if return_piece_num == 1:
        return cut_off_list
    for current_piece_num in sorted(range(2, return_piece_num + 1), reverse=True):
        result_knots_list,iv = combine_helper(data_df, good_name, bad_name, current_piece_num, cut_off_list,True)
        if current_piece_num==2 and iv is not None:
            if iv<0.1:
#                print("sry, there isn't any suitable division for this data set with the column that you give :(")
                return None
        # here we obey the rule that, the function will return the maximum number of bins with maximum IV value, thus if
        # there is no suitable cut_off_list for the current piece, the number of bins will be minus one
        if result_knots_list is not None:
            return result_knots_list,iv
#    print("sry, there isn't any suitable division for this data set with the column that you give :(")
    return None


def combine_tiny_bins1(data_df, good_name, bad_name, max_piece_num, cut_off_list):
    return_piece_num = min(max_piece_num, len(cut_off_list) + 1)
    if return_piece_num == 1:
        return cut_off_list
    for current_piece_num in sorted(range(2, return_piece_num + 1), reverse=True):
        result_knots_list,iv = combine_helper(data_df, good_name, bad_name, current_piece_num, cut_off_list,False)
        if current_piece_num==2 and iv is not None:
            return result_knots_list,iv
        # here we obey the rule that, the function will return the maximum number of bins with maximum IV value, thus if
        # there is no suitable cut_off_list for the current piece, the number of bins will be minus one
        if result_knots_list is not None:
            return result_knots_list,iv
#    print("sry, there isn't any suitable division for this data set with the column that you give :(")
    return None
#campare two mthode of cut:mononie and U.
def combine_tiny_bins(data_df, good_name, bad_name, max_piece_num, cut_off_list):
#    x0=combine_tiny_bins0(data_df, good_name, bad_name, max_piece_num, cut_off_list)
    x1=combine_tiny_bins1(data_df, good_name, bad_name, max_piece_num, cut_off_list)
    if x1 is None:
        print("sry, there isn't any suitable division for this data set with the column that you give :(")
        return [0, len(data_df) - 1]
    else:
        return x1[0]
# this function is used for return the import statistical indicators for the binning result
def important_indicator_calculator(data_df, good_name, bad_name, factor_name, knots_list, na_df):
    if len(na_df) != 0:
        total_good = sum(data_df[good_name]) + sum(na_df[good_name])
        total_bad = sum(data_df[bad_name]) + sum(na_df[bad_name])
        na_good_percent = na_df[good_name] / float(total_good)
        na_bad_percent = na_df[bad_name] / float(total_bad)
        na_indicator = pd.DataFrame({'Bin': list(na_df.ix[:,0]), 'KS': [None]*len(na_df), 'WOE': list(np.log(na_bad_percent/na_good_percent)),
                                     'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                                     'total_count': list(na_df[good_name] + na_df[bad_name]),
                                     'bad_rate': list(na_df[bad_name] /(na_df[good_name] + na_df[bad_name]))})
    else:
        total_good = sum(data_df[good_name])
        total_bad = sum(data_df[bad_name])
        na_indicator = pd.DataFrame()
    default_CDF = np.cumsum(data_df[bad_name]) / total_bad
    undefault_CDF = np.cumsum(data_df[good_name]) / total_good
    ks_list = list(abs(default_CDF - undefault_CDF).loc[knots_list[:len(knots_list) - 1]])
    temp_df_list = []
    bin_list = []
    for i in range(1, len(knots_list)):
        if i == 1:
            temp_df_list.append(data_df.loc[knots_list[i - 1]:knots_list[i]])
            bin_list.append('(-inf, ' + data_df[factor_name][knots_list[i]] + ']')
        else:
            temp_df_list.append(data_df.loc[knots_list[i - 1] + 1:knots_list[i]])
            if i == len(knots_list) - 1:
                bin_list.append('(' + data_df[factor_name][knots_list[i - 1]] + ', inf)')
            else:
                bin_list.append(
                    '(' + data_df[factor_name][knots_list[i - 1]] + ', ' + str(
                        data_df[factor_name][knots_list[i]]) + ']')
    good_percent_series = pd.Series(list(map(lambda x: float(sum(x[good_name])) / total_good, temp_df_list)))
    bad_percent_series = pd.Series(list(map(lambda x: float(sum(x[bad_name])) / total_bad, temp_df_list)))
    woe_list = list(np.log(bad_percent_series/good_percent_series))   
    IV_list = list((good_percent_series - bad_percent_series) * np.log(good_percent_series / bad_percent_series))
    total_list = list(map(lambda x: sum(x[good_name]) + sum(x[bad_name]), temp_df_list))
    bad_rate_list = list(map(lambda x: float(sum(x[bad_name])) / (sum(x[good_name]) + sum(x[bad_name])), temp_df_list))
    non_na_indicator = pd.DataFrame({'Bin': bin_list, 'KS': ks_list, 'WOE': woe_list, 'IV': IV_list,
                                     'total_count': total_list, 'bad_rate': bad_rate_list})
    result_indicator = pd.concat([non_na_indicator, na_indicator], axis=0).reset_index(drop=True)
    return result_indicator


# the most significant difference between the two function is the all_information part
# let's redefine this function
def all_information(data_df, na_df, total_rec, piece, rate, factor_name, bad_name, good_name,not_in_list):
    # to get the final result of the binning part, we need two process: split and merge
    split_knots = new_ks_auto(data_df, total_rec, piece, rate, good_name, bad_name)
    print (split_knots)
    best_knots = combine_tiny_bins(data_df, good_name, bad_name, piece, split_knots)
    if best_knots==[] and (min(data_df[bad_name])>0 and min(data_df[good_name])>0):
        na_df=na_df.append(data_df) 
        not_in_list=not_in_list+list(data_df.ix[:,0])  
    return important_indicator_calculator(data_df, good_name, bad_name, factor_name, best_knots, na_df),not_in_list,best_knots

# here is the final_outer function
def Best_KS_Bin(flag_name, factor_name, data=pd.DataFrame(), bad_name='bad', good_name='good', piece=5, rate=0.05, min_bin_size=50, not_in_list=[]):
    # print(time.localtime(time.time()))
    # in order to avoid revising the raw data, we choose to copy the current data and contain only the factor and flag
    # columns
    if len(data) == 0:
        print ('Error: there is no data')
        return pd.DataFrame()
    work_data = data.loc[data.index, [factor_name, flag_name]]
    # since the data without flag is meaningless thus, we can use the helper function to help us filer these data
    work_data = verify_df_two(work_data, flag_name)
    if len(work_data) == 0:
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
        return pd.DataFrame(),not_in_list
    # total_good = sum(non_na_df[good_name]) + sum(na_df[good_name])
    # total_bad = sum(non_na_df[bad_name]) + sum(na_df[bad_name])
    # total_rec = total_good + total_bad
    total_rec = len(work_data)
    min_bin_size_rate = min_bin_size/float(total_rec)
    min_bin_size_r = max(rate, min_bin_size_rate)
    result,not_in_list,best_knots = all_information(non_na_df, na_df, total_rec, piece, min_bin_size_r, factor_name, bad_name, good_name,not_in_list)
    # print(time.localtime(time.time()))
    if len(best_knots)==2:
        print('sry, there are no data available for separate process :(')
        return pd.DataFrame(),not_in_list
    return result,not_in_list
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
  
    
    
def df_woe(flag_name, data=pd.DataFrame(), data1=pd.DataFrame(),bad_name='bad', good_name='good', con_rate = 0.90, piece=5, rate=0.05, min_bin_size=50, not_in_list=[], not_var_list=[], flag_var_list=[]):
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
    if len(data1)>0:
        data_woe1 = data1[flag_var_list]
        
    data_bin = pd.DataFrame()
    if len(data) == 0:
        print ('Original input data is empty')
        return pd.DataFrame()
    var_list = data.columns
    not_var_list.extend([flag_name])
    not_var_list.extend(not_in_list)
    not_var_list.append('time_stamp')
    not_in_list.extend(['None', 'nan'])
    not_max_var = []
    for var in data.columns:
        percent = data[var].value_counts(normalize=True, dropna=False)
        #判定单一值比率
        if percent.max() >= con_rate:
            not_max_var.append(var)
    target_var_list = list(set(var_list) - set(not_var_list)-set(not_max_var))
#    target_var_list=col        
    iv_list = []
    ks_list = []
    target_var_list1=[]

    if len(target_var_list) == 0:
        print ('No variable available for analysis')
        return pd.DataFrame()
    iter = 0
    for var in target_var_list: 
        
        print (var)
        try:
            var_stat,not_in_list_1= Best_KS_Bin(flag_name, var, data, bad_name, good_name, piece, rate, min_bin_size, not_in_list)
            if len(var_stat) > 0:
                if len(var_stat['WOE']) != len(set(var_stat['WOE'])):
                    var_stat.ix[var_stat['Bin']=='NA','WOE'] = var_stat.ix[var_stat['Bin']=='NA','WOE']+0.0000001
                var_stat['var'] = var
                
                var_stat['WOE']=var_stat[['total_count','WOE']].apply(lambda x: 0 if x[0]<len(data)*0.05 else x[1],axis=1)            
                bin_dic = dict(zip(var_stat['WOE'], var_stat['Bin']))
                for woe in bin_dic:
                    match_case = re.compile("\(|\)|\[|\]")
                    end_points = match_case.sub('', bin_dic[woe]).split(', ')
                    bin_dic[woe] = end_points
                data_woe[var] = list(map(lambda x: var_woe(x, bin_dic, not_in_list_1), data[var].map(lambda x: float(x))))
                if len(data1)>0:   
                    data_woe1[var] = list(map(lambda x: var_woe(x, bin_dic, not_in_list_1), data1[var].map(lambda x: float(x))))
                ivv=list(var_stat['IV'])
                while float('inf') in ivv:
                    ivv.remove(float('inf'))
                iv = sum(ivv)
                ks = max(var_stat['KS'])
                data_bin = pd.concat([data_bin, var_stat])
                # info_dic.update({var: [iv, ks]})
                iv_list.append(iv)
                ks_list.append(ks)
                iter += 1
                print (iter)
                target_var_list1.append(var)
            else:
    #                iv_list.append('nan')
    #                ks_list.append('nan')
                print (var, ' Should be checked')
        except:
            print (var+'--error')
            pass

    data_stat = pd.DataFrame({'var': target_var_list1, 'iv': iv_list, 'ks': ks_list}).sort_values(by='iv', ascending=False)
    if len(data1)>0:
        data_woe.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        data_woe1.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        return data_woe,data_woe1, data_bin, data_stat
    else:
        data_woe.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu',np.nan],0,inplace=True)
        return data_woe, data_bin, data_stat



#Best_KS_Bin(flag_name, factor_name, xx, bad_name='bad', good_name='good', piece=5, rate=0.05, min_bin_size=50, not_in_list=[])
#    
#df_woe(flag_name, data, bad_name='bad', good_name='good', con_rate = 0.90, piece=5, rate=0.05, min_bin_size=50, not_in_list=[], not_var_list=[], flag_var_list=[])

class f_bin_woe:
    def __init__(self): 
        self.flag=''
        self.model=pd.DataFrame()
        self.col=[]
        self.cor_method='pearson'
        self.p_cri=0.5
        self.ivv=pd.DataFrame({'v':[],'iv':[],'i_index':[],'woe_t':[]})
        self.id=''
        self.ce=[]
        self.obj=pd.DataFrame({'n':[],'obj':[],'re':[]})
        self.N=5
        self.coo=[]

    def load(self,df,a,b,c,d):
        a=copy.copy(a)
        b=copy.copy(b)
        c=copy.copy(c)
        if b in a:
            a.remove(b)
        if c in a:
            a.remove(c)            
        self.model=df[a+[b]+[c]]
        self.flag=b
        self.col=list(set(a))        
        self.id=c
        self.model.replace(['non','none','None','NONE','null','NULL','Null','"null"','[]','[ ]','{}','{ }',' ','nu'],np.nan,inplace=True)
        self.N=d
        
#计算iv最大的woe 
        
    def df_dis(self,df,y):
        df=copy.copy(df)
        for x in self.col:
            if df[x].dtype=='object':
                df_agg = Cal_WOE(df,x, y)
                ob=pd.DataFrame({'n':[x],'obj':[list(df_agg.index)],'re':[list(df_agg[x + '_woe'])]})
                self.obj=self.obj.append(ob,ignore_index=True)
                df[x] = df[x].replace(df_agg.index, df_agg[x + '_woe'])
        return df



def dwoe(m,dddf2,col,not_in_list=[]):
    for k in col:
        cd=dddf2[dddf2['var']==k].copy()
        cd['1']=range(len(cd))
        cd['WOE']=cd['WOE']+cd['1']/100000
        dc=dict()
        for l in range(len(cd)):                
            dc[list(cd['WOE'])[l]]=cd['Bin'][l].replace(')','').replace('(','').replace(']','').split(',')
            if len(dc[list(cd['WOE'])[l]])==1:
                dc[list(cd['WOE'])[l]]=dc[list(cd['WOE'])[l]][0]
        for woe in dc:
            if len(dc[woe])!=2:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', dc[woe]).split(', ')
                dc[woe] = end_points
            
        for l in dc:
            if len(dc[l])==1:
                not_in_list.append(dc[l][0])

        m[k]= list(map(lambda x: var_woe(x, dc, not_in_list), m[k].map(lambda x: float(x))))
    return m



def dwoe1(m,dddf2,col,not_in_list=[]):
    for k in col:
        cd=dddf2[dddf2['var']==k].copy()
        cd['1']=range(len(cd))
        cd['WOE']=cd['WOE']
        dc=dict()
        for l in range(len(cd)):                
            dc[list(cd['WOE'])[l]]=cd['Bin'][l].replace(')','').replace('(','').replace(']','').split(',')
            if len(dc[list(cd['WOE'])[l]])==1:
                dc[list(cd['WOE'])[l]]=dc[list(cd['WOE'])[l]][0]
        for woe in dc:
            if len(dc[woe])!=2:
                match_case = re.compile("\(|\)|\[|\]")
                end_points = match_case.sub('', dc[woe]).split(', ')
                dc[woe] = end_points
            
        for l in dc:
            if len(dc[l])==1:
                not_in_list.append(dc[l][0])

        m[k]= list(map(lambda x: var_woe(x, dc, not_in_list), m[k].map(lambda x: float(x))))
    return m



#def vvar_woe(x, bin_dic, not_in_list):
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

def woe_cal(m,col,y,bad_name,good_name):
    iv=[]
    for fac in col:
        na_df = group_by_df(m, y, fac, bad_name, good_name, False)
        total_good = sum(na_df[good_name])
        total_bad = sum(na_df[bad_name])
        na_good_percent = na_df[good_name] / float(total_good)
        na_bad_percent = na_df[bad_name] / float(total_bad)
        na_indicator = pd.DataFrame({'Bin': list(na_df.ix[:,0]), 'KS': [None]*len(na_df), 'WOE': list(np.log(na_bad_percent/na_good_percent)),
                                     'IV': list((na_good_percent - na_bad_percent) * np.log(na_good_percent / na_bad_percent)),
                                     'total_count': list(na_df[good_name] + na_df[bad_name]),
                                     'bad_rate': list(na_df[bad_name] /(na_df[good_name] + na_df[bad_name]))})    
        na_indicator['var'] = fac
        iv.append(na_indicator['IV'].sum())
    return pd.DataFrame({'var':col,'iv':iv})
    



    


