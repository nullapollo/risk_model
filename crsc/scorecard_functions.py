import numpy as np
import pandas as pd


def SplitData(df, col, numOfSplit, special_attribute=[]):
    """
    计算特征 col 在指定分箱数 numOfSplit 下的切分点值，排除特殊取值 special_attribute

    :param df: 按照col排序后的数据集
    :param col: 待分箱的特征
    :param numOfSplit: 切分的组别数
    :param special_attribute: 在切分数据集的时候，某些特殊值需要排除在外

    :return: 返回特征 col 在指定分箱数下的切分点值
    """
    df2 = df.copy()
    if special_attribute != []:
        df2 = df.loc[~df[col].isin(special_attribute)]
    N = df2.shape[0]
    n = int(N / numOfSplit)
    splitPointIndex = [i * n for i in range(1, numOfSplit)]
    rawValues = sorted(list(df2[col]))
    splitPoint = [rawValues[i] for i in splitPointIndex]
    splitPoint = sorted(list(set(splitPoint)))
    return splitPoint


def UnsupervisedSplitBin(df, var, numOfSplit=5, method='equal freq'):
    '''
    :param df: 数据集
    :param var: 需要分箱的变量。仅限数值型。
    :param numOfSplit: 需要分箱个数，默认是5
    :param method: 分箱方法，'equal freq'：，默认是等频，否则是等距
    :return: 切分点
    '''
    if method == 'equal freq':
        N = df.shape[0]
        n = N / numOfSplit
        splitPointIndex = [i * n for i in range(1, numOfSplit)]
        rawValues = sorted(list(df[col]))
        splitPoint = [rawValues[i] for i in splitPointIndex]
        splitPoint = sorted(list(set(splitPoint)))
        return splitPoint
    else:
        var_max, var_min = max(df[var]), min(df[var])
        interval_len = (var_max - var_min)*1.0/numOfSplit
        splitPoint = [var_min + i*interval_len for i in range(1,numOfSplit)]
        return splitPoint


def AssignGroup(x, bin):
    """
    :param x: 某个变量的某个取值
    :param bin: 上述变量的分箱结果（连续变量的切点）
    :return: x在分箱结果下的映射
    """
    N = len(bin)
    if x <= min(bin):
        return min(bin)
    elif x > max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


def AssignBin(x, cutOffPoints, special_attribute=[]):
    """
    :param x: 某个变量的某个取值
    :param cutOffPoints: 上述变量的分箱结果，用切分点表示
    :param special_attribute:  不参与分箱的特殊取值
    :return: 分箱后的对应的第几个箱，从0开始
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3


    1. 当 special_attribute 不空时，总的最小分箱数 totalNumBin = len(special_attribute) + 1 （正常值至少被分为1箱，cutOffPoints 有空的可能）
    2. 当 special_attribute 为空时，总的最小分箱数 totalNumBin = 2 (正常值至少分两箱，连续变量会有一个切点，cutOffPoints 不会空）

    """
    # 对于正常取值的分箱数，在 cutOffPoints 列表为空时，所有正常值分为同1组
    numBin = len(cutOffPoints) + 1

    # 特殊取值部分，按照特殊值的个数，分为相应的组数，记录为负值分箱编号
    if x in special_attribute:
        i = special_attribute.index(x) + 1
        return 'Bin {}'.format(0-i)

    if x <= cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0, numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


def Chi2(df, total_col, bad_col, overallRate):
    """
    计算特征
    :param df: 包含全部样本总计与坏样本总计的数据框
    :param total_col: 全部样本的个数
    :param bad_col: 坏样本的个数
    :param overallRate: 全体样本的坏样本占比

    :return: 卡方值
    """
    df2 = df.copy()
    # 期望坏样本个数＝全部样本个数*平均坏样本占比
    df2['expected'] = df[total_col].apply(lambda x: x * overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2


def BinBadRate(df, col, target, grantRateIndicator=0):
    '''
    计算特征 col 在取值下的 bad_rate

    :param df: 需要计算正负样本比率的数据集
    :param col: 需要计算正负样本比率的特征（X)
    :param target: 目标变量标签（Y）
    :param grantRateIndicator: 1返回总体的坏样本率，0不返回

    :return: 每箱的坏样本率，以及总体的坏样本率（当grantRateIndicator＝＝1时）
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad * 1.0 / x.total, axis=1)
    dicts = dict(zip(regroup[col],regroup['bad_rate']))
    if grantRateIndicator==0:
        return (dicts, regroup)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    overallRate = B * 1.0 / N
    return (dicts, regroup, overallRate)


def ChiMerge(df, col, target, max_interval=5, special_attribute=[], minBinPcnt=0, monotone_check=False):
    """
    对特征 col 用卡方分箱法进行分箱，选取最佳切点，默认的最大分箱数为5
    - 卡方分箱法更为客观，通过计算两组客群的卡方值来判断是否适合进行合并
    - 卡方值较大，说明客群间具备显著的差异，每次选择最小卡方值的组进行合并
    - 卡方分箱是个自下而上的聚合过程，初始为较多分箱数，逐步聚合成较少的分箱数
    - 卡方分箱法聚合过程中，只有卡方值作为标准，并未考虑组间bad rate的单调性


    :param df: 包含目标变量与分箱属性的数据框
    :param col: 需要分箱的属性
    :param target: 目标变量，取值0或1
    :param max_interval: 最大分箱数，如果原始属性的取值个数低于该参数，不执行这段函数（指正常值的分箱数，默认为5箱）
    :param special_attribute: 不参与分箱的属性取值，只做数据提出
    :param minBinPcnt：最小箱的占比，默认为0

    :return: 分箱结果
    """

    # 计算特征 col 的正常取值水平
    if len(special_attribute) >= 1:   # 存在不做分箱处理的变量取值（如，特殊值标记，缺失值标记，单独分为一类，这里直接剔除）
        df2 = df.loc[~df[col].isin(special_attribute)]
    else:
        df2 = df.copy()
    # 已经是剔除掉异常值后的取值集合
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)

    if N_distinct <= max_interval:
        #如果原始属性的取值个数低于max_interval，不执行这段函数
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1] + special_attribute
    else:
        # 原始正常取值大于 max_interval，进行合并计算

        # *********************************************************************************************
        # 步骤一: 通过col对数据集进行分组，求出每组的总样本数与坏样本数
        if N_distinct > 100:
            split_x = SplitData(df2, col, 100)
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df2[col]
        # 总体bad rate将被用来计算expected bad count
        (binBadRate, regroup, overallRate) = BinBadRate(df2, 'temp', target, grantRateIndicator=1)

        # 首先，每个单独的属性值将被分为单独的一组
        # 对属性值进行排序，然后两两组别进行合并
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]

        # *********************************************************************************************
        # 步骤二：建立循环，不断合并最优的相邻两个组别，直到最终分裂出来的分箱数<＝预设的最大分箱数
        split_intervals = max_interval
        while (len(groupIntervals) > split_intervals):  # 终止条件: 当前分箱数＝预设的分箱数
            # 每次循环时, 计算合并相邻组别后的卡方值；具有最小卡方值的合并方案，是最优方案
            chisqList = []
            for k in range(len(groupIntervals)-1): # 逐个两两合并，计算卡方值
                temp_group = groupIntervals[k] + groupIntervals[k+1]
                df2b = regroup.loc[regroup['temp'].isin(temp_group)]
                chisq = Chi2(df2b, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            best_comnbined = chisqList.index(min(chisqList))
            groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[best_comnbined])
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        # *********************************************************************************************
        # 步骤三：检查是否有箱没有好或者坏样本。如果有，需要跟相邻的箱进行合并，直到每箱同时包含好坏样本
        groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
        df2['temp_Bin'] = groupedvalues
        (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
        [minBadRate, maxBadRate] = [min(binBadRate.values()),max(binBadRate.values())]
        while minBadRate == 0 or maxBadRate == 1:
            # 找出全部为好／坏样本的箱
            indexForBad01 = regroup[regroup['bad_rate'].isin([0, 1])].temp_Bin.tolist()
            bin=indexForBad01[0] # 循环的在判断，所以这里只取一个也可以，后面会循环到的
            # 如果是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
            if bin == max(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[:-1]
            # 如果是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
            elif bin == min(regroup.temp_Bin):
                cutOffPoints = cutOffPoints[1:]
            # 如果是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
            else:
                # 和前一箱进行合并，并且计算卡方值
                currentIndex = list(regroup.temp_Bin).index(bin)
                prevIndex = list(regroup.temp_Bin)[currentIndex - 1]
                df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                chisq1 = Chi2(df2b, 'total', 'bad', overallRate)

                # 和后一箱进行合并，并且计算卡方值
                laterIndex = list(regroup.temp_Bin)[currentIndex + 1]
                df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, bin])]
                (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                chisq2 = Chi2(df2b, 'total', 'bad', overallRate)

                if chisq1 < chisq2:
                    cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                else:
                    cutOffPoints.remove(cutOffPoints[currentIndex])

            # 完成合并之后，需要再次计算新的分箱准则下，每箱是否同时包含好坏样本
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            (binBadRate, regroup) = BinBadRate(df2, 'temp_Bin', target)
            [minBadRate, maxBadRate] = [min(binBadRate.values()), max(binBadRate.values())]

        # *********************************************************************************************
        # 步骤四：需要检查分箱后的最小箱占比是否高于阈值，否则要进行合并
        if minBinPcnt > 0:
            groupedvalues = df2['temp'].apply(lambda x: AssignBin(x, cutOffPoints))
            df2['temp_Bin'] = groupedvalues
            valueCounts = groupedvalues.value_counts().to_frame()
            valueCounts['pcnt'] = valueCounts['temp'].apply(lambda x: x * 1.0 / len(df2['temp']))
            valueCounts = valueCounts.sort_index()
            minPcnt = min(valueCounts['pcnt'])
            while minPcnt < minBinPcnt and len(cutOffPoints) > 2:
                # 找出占比最小的箱
                indexForMinPcnt = valueCounts[valueCounts['pcnt'] == minPcnt].index.tolist()[0]
                # 如果占比最小的箱是最后一箱，则需要和上一个箱进行合并，也就意味着分裂点cutOffPoints中的最后一个需要移除
                if indexForMinPcnt == max(valueCounts.index):
                    cutOffPoints = cutOffPoints[:-1]
                # 如果占比最小的箱是第一箱，则需要和下一个箱进行合并，也就意味着分裂点cutOffPoints中的第一个需要移除
                elif indexForMinPcnt == min(valueCounts.index):
                    cutOffPoints = cutOffPoints[1:]
                # 如果占比最小的箱是中间的某一箱，则需要和前后中的一个箱进行合并，依据是较小的卡方值
                else:
                    # 和前一箱进行合并，并且计算卡方值
                    currentIndex = list(valueCounts.index).index(indexForMinPcnt)
                    prevIndex = list(valueCounts.index)[currentIndex - 1]
                    df3 = df2.loc[df2['temp_Bin'].isin([prevIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3, 'temp_Bin', target)
                    chisq1 = Chi2(df2b, 'total', 'bad', overallRate)
                    # 和后一箱进行合并，并且计算卡方值
                    laterIndex = list(valueCounts.index)[currentIndex + 1]
                    df3b = df2.loc[df2['temp_Bin'].isin([laterIndex, indexForMinPcnt])]
                    (binBadRate, df2b) = BinBadRate(df3b, 'temp_Bin', target)
                    chisq2 = Chi2(df2b, 'total', 'bad', overallRate)
                    if chisq1 < chisq2:
                        cutOffPoints.remove(cutOffPoints[currentIndex - 1])
                    else:
                        cutOffPoints.remove(cutOffPoints[currentIndex])

        # *********************************************************************************************
        # 步骤五：单调性检查，不满足单调性时，进一步合并
        # while True:
        #     # 每次循环时, 计算合并相邻组别后的卡方值；具有最小卡方值的合并方案，是最优方案
        #     chisqList = []
        #     for k in range(len(groupIntervals)-1):
        #         temp_group = groupIntervals[k] + groupIntervals[k+1]
        #         df2b = regroup.loc[regroup['temp'].isin(temp_group)]
        #         chisq = Chi2(df2b, 'total', 'bad', overallRate)
        #         chisqList.append(chisq)
        #     best_comnbined = chisqList.index(min(chisqList))
        #     groupIntervals[best_comnbined] = groupIntervals[best_comnbined] + groupIntervals[best_comnbined+1]
        #     # after combining two intervals, we need to remove one of them
        #     groupIntervals.remove(groupIntervals[best_comnbined])
        #
        #     if len(groupIntervals) <= split_intervals: # 达到指定的分箱个数时，才开始进行单调性检查
        #         if monotone_check:
        #             # 单调性检查
        #             tempIntervals = [sorted(i) for i in groupIntervals]
        #             cutOffPoints = [max(i) for i in tempIntervals[:-1]]
        #             df2[col+'_Bin'] = df2[col].map(lambda x: AssignBin(x, cutOffPoints, special_attribute=[]))
        #             monotone = BadRateMonotone(df2, col+'_Bin', 'y', special_attribute=[])   # 检验分箱后的单调性是否满足
        #             if monotone or len(groupIntervals) == 2: # 一旦达到单调或者只分两箱
        #                 break
        #         else:
        #             # 无需单调性检查时，分箱数达到要求，直接退出合并计算循环
        #             break
        # groupIntervals = [sorted(i) for i in groupIntervals]
        # cutOffPoints = [max(i) for i in groupIntervals[:-1]]

        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints


def BadRateEncoding(df, col, target):
    '''
    对于取值数大于5的变量，采用bad rate进行编码，按照连续型变量的方式进行处理

    :param df: dataframe containing feature and target
    :param col: the feature that needs to be encoded with bad rate, usually categorical type
    :param target: good/bad indicator

    :return: the assigned bad rate to encode the categorical feature
    '''
    regroup = BinBadRate(df, col, target, grantRateIndicator=0)[1]
    """
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    """
    br_dict = dict(zip(regroup[col], regroup['bad_rate']))
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'bad_rate':br_dict}


def CalcWOE(df, col, target):
    """
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
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: -np.log(x.good_pcnt*1.0/x.bad_pcnt), axis=1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}


## 判断某变量的坏样本率是否单调
def BadRateMonotone(df, sortByVar, target, special_attribute=[]):
    '''
    :param df: 包含检验坏样本率的变量，和目标变量
    :param sortByVar: 需要检验坏样本率的变量
    :param target: 目标变量，0、1表示好、坏
    :param special_attribute: 不参与检验的特殊值
    :return: 坏样本率单调与否
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    if len(set(df2[sortByVar])) <= 2:
        return True
    regroup = BinBadRate(df2, sortByVar, target)[1]
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    # 求值按照先 and 后 or 来做，所以可以判断单调性，这里 -1 不参加单调性评估，所以从 1 开始
    badRateNotMonotone = [badRate[i]<badRate[i+1] and badRate[i] < badRate[i-1] or badRate[i]>badRate[i+1] and badRate[i] > badRate[i-1]
                       for i in range(1, len(badRate)-1)]
    if True in badRateNotMonotone:
        return False
    else:
        return True


def MergeBad0(df, col, target, direction='bad'):
    """
    计算特征 col 的分箱合并方案

    :param df: 包含检验0％或者100%坏样本率
    :param col: 分箱后的变量或者类别型变量。检验其中是否有一组或者多组没有坏样本或者没有好样本。如果是，则需要进行合并
    :param target: 目标变量，0、1表示好、坏

    :return: 合并方案，使得每个组里同时包含好坏样本
    """
    regroup = BinBadRate(df, col, target)[1]
    if direction == 'bad':
        # 如果是合并0坏样本率的组，则跟最小的非0坏样本率的组进行合并
        regroup.sort_values(by='bad_rate', inplace=True)
    else:
        # 如果是合并0好样本率的组，则跟最小的非0好样本率的组进行合并
        regroup.sort_values(by='bad_rate', ascending=False, inplace=True)
    regroup.index = range(regroup.shape[0])
    
    # 合并需要重组的箱，先合并再删除
    col_regroup = [[i] for i in regroup[col]]
    del_index = []
    for i in range(regroup.shape[0]-1):
        col_regroup[i+1] = col_regroup[i] + col_regroup[i+1]
        del_index.append(i)
        if direction == 'bad':
            if regroup['bad_rate'][i+1] > 0:
                break
        else:
            if regroup['bad_rate'][i+1] < 1:
                break
    col_regroup2 = [col_regroup[i] for i in range(len(col_regroup)) if i not in del_index]
    newGroup = {}
    for i in range(len(col_regroup2)):
        for g2 in col_regroup2[i]:
            newGroup[g2] = 'Bin '+ str(i)
    return newGroup