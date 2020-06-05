from crsc.bins.cart_bins import get_bestsplit_list
import pandas as pd
import numpy as np


def get_woeinfo_by_group(df, col, target):
    """
    按照分箱结果进行 woe 和 iv 统计
    group/total/bad/good/bad_rate/good_rate/total_pcnt/bad_pcnt/good_pcnt/woe/iv/sum_iv

    :param df: 包含 bin 和 y 的数据集 pandas.dataframe
    :param col: bin 的字段名称 string, 分箱的标签
    :param target: y 的字段名称 string, 取值 0/1 数值
    :return: woe 和 iv 的统计结果 pandas.dataframe
    """

    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})

    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})

    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['good'] = regroup['total'] - regroup['bad']
    regroup['bad_rate'] = regroup['bad'] / regroup['total']
    regroup['good_rate'] = regroup['good'] / regroup['total']

    # 这里输入之前，都要确认清楚， N/B/G 都不会是 0
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    G = N - B

    regroup['total_pcnt'] = 0 if N == 0 else regroup['total'] / N

    regroup['bad_pcnt'] = regroup['bad'].apply(lambda x: 0 if B == 0 else x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].apply(lambda x: 0 if G == 0 else x * 1.0 / G)
    regroup['bad_pcnt'].replace(0, 1e-6, inplace=True)
    regroup['good_pcnt'].replace(0, 1e-6, inplace=True)

    regroup['woe'] = regroup.apply(lambda x: np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)
    regroup['iv'] = regroup.apply(lambda x: (x.bad_pcnt - x.good_pcnt) * np.log(x.bad_pcnt * 1.0 / x.good_pcnt), axis=1)

    regroup['sum_iv'] = sum(regroup['iv'])

    regroup['col'] = col
    regroup.rename(columns={col: 'group'}, inplace=True)

    return regroup


def get_woeinfo_withoutgroup(df_data, col, target, min_rate=0.1):
    """
    对变量进行分箱，再做 woe 和 iv 计算

    :param df_data: 包含 x 和 y 的数据集 pandas.dataframe
    :param col: x 变量名称 string
    :param target: y 目标 string， 取值 0/1
    :param min_rate: 分箱最小箱的比率
    :return: 分箱结果统计 pandas.dataframe
    """

    df_group = pd.DataFrame()
    df_group[target] = df_data[target]

    # 获取分组
    if df_data[col].dtype == 'object':  # 字符型变量，这里直接按
        df_group[col] = df_data[col]
        print('%s is string' % col)

    elif len(df_data[col].unique()) <= 8:  # 数值型变量，取值水平数不超过 8 的也直接按取值分箱
        df_group[col] = df_data[col]
        print('%s is number but unique <=8' % col)

    else:  # 数值型变量，取值水平数超过8的，进行分箱操作
        arr_split = get_bestsplit_list(df_data, col, target, min_rate)
        arr_split.append(min(df_data[col]) - 0.001)
        arr_split.append(max(df_data[col]))
        arr_split = sorted(arr_split)
        df_group[col] = pd.cut(df_data[col], arr_split).tolist()
        print('%s is number and bins' % col)

    # 获取 woe、iv
    return get_woeinfo_by_group(df_group, col, target)


# 先进行分箱，分箱完进行分组，分组完计算woe、iv，计算完进行数据woe转换
def get_woedatas_withoutgroup(df_data, arr_col, target, min_rate=0.1):
    """
    针对一批变量，进行分箱再计算 woe 和 iv 信息

    :param df_data: 包含 x 和 y 的 pandas.dataframe
    :param arr_col: x 的列表
    :param target: y
    :param min_rate: 最小分箱占比
    :return:
    """
    # 获取分组
    df_group = pd.DataFrame()
    df_group[target] = df_data[target]

    for col in arr_col:
        if df_data[col].dtype == 'object':
            df_group[col] = df_data[col]
            print('%s is string' % col)

        elif len(df_data[col].unique()) <= 8:
            df_group[col] = df_data[col]
            print('%s is number but unique <=8' % col)
        else:
            arr_split = get_bestsplit_list(df_data, col, target, min_rate)
            arr_split.append(min(df_data[col]) - 0.001)
            arr_split.append(max(df_data[col]))
            arr_split = sorted(arr_split)
            df_group[col] = pd.cut(df_data[col], arr_split).tolist()
            print('%s is number and bins' % col)

    # 获取 woe、iv
    df_woeinfos = get_woeinfo_df()
    for col in arr_col:
        col_woe = col + '_woe'
        if col_woe in df_group.columns:
            df_group.drop(col_woe, axis=1, inplace=True)
            print('drop %s' % col_woe)
        df_col_woe = get_woeinfo_by_group(df_group, col, target)
        df_woeinfos = pd.concat([df_woeinfos, df_col_woe], axis=0)
        df_col_woe = df_col_woe[['group', 'woe']]
        df_col_woe.rename(columns={'group': col, 'woe': col_woe}, inplace=True)
        # print(df_col_woe)
        df_group = df_group.merge(df_col_woe, how='left', on=col)

    # 获取 woe 数据
    arr_woe = [col for col in df_group.columns if '_woe' in col]
    arr_woe = [target] + arr_woe

    df_woedata = pd.DataFrame()
    df_woedata = df_group[arr_woe]

    return df_woedata, df_woeinfos


def get_woes_details(df_data, arr_col, target):
    # 获取分组
    df_group = pd.DataFrame()
    df_group[target] = df_data[target]
    for col in arr_col:
        if df_data[col].dtype == 'object':
            df_group[col] = df_data[col]
            print('%s is string' % col)

        elif len(df_data[col].unique()) <= 8:
            df_group[col] = df_data[col]
            print('%s is number but unique <=8' % col)
        else:
            arr_split = get_bestsplit_list(df_data, col, target, min_rate=0.1)
            arr_split.append(min(df_data[col]) - 0.001)
            arr_split.append(max(df_data[col]))
            arr_split = sorted(arr_split)
            df_group[col] = pd.cut(df_data[col], arr_split).tolist()
            print('%s is number and bins' % col)

    # 获取 woe、iv
    df_woe = pd.DataFrame()
    for col in arr_col:
        col_woe = col + '_woe'
        if (col_woe in df_group.columns):
            df_group.drop(col_woe, axis=1, inplace=True)
            print('drop %s' % col_woe)
        df_col_woe = get_woe(df_group, col, target)
        df_col_woe = df_col_woe[['group', 'woe']]
        df_col_woe.rename(columns={'group': col, 'woe': col_woe}, inplace=True)
        # print(df_col_woe)
        df_group = df_group.merge(df_col_woe, how='left', on=col)

    arr_woe = [col for col in df_group.columns if '_woe' in col]
    arr_woe = arr_woe + [target]
    df_woe = df_group[arr_woe]

    return df_woe


def get_woeinfo_df():
    return pd.DataFrame(
        columns=['group', 'total', 'bad', 'good', 'bad_pcnt', 'good_pcnt', 'woe', 'iv', 'sum_iv', 'bad_rate',
                 'good_rate', 'col', 'total_pcnt'])
