import pandas as pd
import numpy as np


def woe_iv2(df, col, target):
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
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['bad_pcnt'].replace(0, 0.000001, inplace=True)
    regroup['good_pcnt'].replace(0, 0.000001, inplace=True)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    regroup['IV'] = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    # regroup.rename(columns={col:'group'},inplace=True)
    return regroup


def woe_iv(df, col, target):
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
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x * 1.0 / B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['bad_pcnt'].replace(0, 0.000001, inplace=True)
    regroup['good_pcnt'].replace(0, 0.000001, inplace=True)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    regroup['IV'] = regroup.apply(lambda x: (x.good_pcnt - x.bad_pcnt) * np.log(x.good_pcnt * 1.0 / x.bad_pcnt), axis=1)
    regroup.rename(columns={col: 'group'}, inplace=True)
    return regroup


def calc_score_median(sample_set, var):
    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) - 1):
        var_median = (var_list[i] + var_list[i + 1]) / 2
        var_median_list.append(var_median)
    return var_median_list


def choose_best_split(sample_set, var, min_sample):
    score_median_list = calc_score_median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set['target'])
    sample0_cnt = sample_cnt - sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

    bestGini = 0.0
    bestSplit_point = 0.0
    bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]

        left_cnt = left.shape[0];
        right_cnt = right.shape[0]
        left1_cnt = sum(left['target'])
        right1_cnt = sum(right['target'])
        left0_cnt = left_cnt - left1_cnt
        right0_cnt = right_cnt - right1_cnt
        left_ratio = left_cnt / sample_cnt
        right_ratio = right_cnt / sample_cnt

        if left_cnt < min_sample or right_cnt < min_sample:
            continue

        Gini_left = 1 - np.square(left1_cnt / left_cnt) - np.square(left0_cnt / left_cnt)
        Gini_right = 1 - np.square(right1_cnt / right_cnt) - np.square(right0_cnt / right_cnt)
        Gini_temp = Gini - (left_ratio * Gini_left + right_ratio * Gini_right)
        if Gini_temp > bestGini:
            bestGini = Gini_temp
            bestSplit_point = score_median_list[i]
            if median_len > 1:
                bestSplit_position = i / (median_len - 1)
            else:
                bestSplit_position = i / median_len
        else:
            continue
    Gini = Gini - bestGini
    return bestSplit_point, bestSplit_position


def bining_data_split(sample_set, var, min_sample, split_list):
    split, position = choose_best_split(sample_set, var, min_sample)
    if split != 0.0:
        split_list.append(split)
    sample_set_left = sample_set[sample_set[var] < split]
    sample_set_right = sample_set[sample_set[var] > split]
    if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_left, var, min_sample, split_list)
    else:
        None
    if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_right, var, min_sample, split_list)
    else:
        None


def get_bestsplit_list(sample_set, var):
    min_df = sample_set.shape[0] * 0.05
    split_list = []
    bining_data_split(sample_set, var, min_df, split_list)
    return split_list


def get_woe_data(df0, df, cols):
    for col in cols:
        try:
            if df0[col].dtype == 'object':
                if df0[col].nunique() < 100:
                    df0[col] = df0[col].fillna('0')
                    df0[col] = df0[col].replace({None: '0'})
                    df_woe = woe_iv(df0, col, 'target')
                    df_woe['sum_iv'] = sum(df_woe['IV'])
                    df_woe['col'] = col
                    print(df_woe)
                    print(col, 'IV:', sum(df_woe['IV']))
                    df = pd.concat([df, df_woe], axis=0)
                else:
                    continue
            elif len(df0[col].unique()) < 6:
                print(col, df0[col].unique())
                df0[col] = df0[col].fillna(0)
                df_woe = woe_iv(df0, col, 'target')
                df_woe['sum_iv'] = sum(df_woe['IV'])
                df_woe['col'] = col
                print(df_woe)
                print(col, 'IV:', sum(df_woe['IV']))
                df = pd.concat([df, df_woe], axis=0)
            else:
                df0[col] = df0[col].fillna(0)
                df_cart = get_bestsplit_list(df0, col)
                df_cart.append(min(df0[col]) - 100)
                df_cart.append(max(df0[col]) + 100)
                df_sort = sorted(df_cart)
                print(col, df_sort)
                colg = col + '_g'
                df0[colg] = pd.cut(df0[col], df_sort)
                df_woe = woe_iv(df0, colg, 'target')
                # df_woe.drop(['bad_pcnt', 'good_pcnt'], axis=1, inplace=True)
                df_woe['sum_iv'] = sum(df_woe['IV'])
                df_woe['col'] = col
                print(df_woe)
                print(col, 'IV:', sum(df_woe['IV']))
                df = pd.concat([df, df_woe], axis=0)
                df0.drop(colg, axis=1, inplace=True)
        except:
            continue
    return df


def get_woe(df0, cols):
    for col in cols:
        if df0[col].dtype == 'object':
            if df0[col].nunique() < 100:
                df0[col] = df0[col].fillna('0')
                df0[col] = df0[col].replace({None: '0'})
                df_woe = woe_iv2(df0, col, 'target')
                df_woe['sum_iv'] = sum(df_woe['IV'])
                df_woe['col'] = col
                print(df_woe)
                print(col, 'IV:', sum(df_woe['IV']))
                df_woe2 = df_woe[[col, 'WOE']]
                df_woe2.rename(columns={'WOE': col + '_woe'}, inplace=True)
                df0 = pd.merge(df0, df_woe2, how='left', on=col)
            else:
                continue
        elif len(df0[col].unique()) < 7:
            print(col, df0[col].unique())
            df0[col] = df0[col].fillna(0)
            df_woe = woe_iv2(df0, col, 'target')
            df_woe['sum_iv'] = sum(df_woe['IV'])
            df_woe['col'] = col
            print(df_woe)
            print(col, 'IV:', sum(df_woe['IV']))
            df_woe2 = df_woe[[col, 'WOE']]
            df_woe2.rename(columns={'WOE': col + '_woe'}, inplace=True)
            df0 = pd.merge(df0, df_woe2, how='left', on=col)
        else:
            df0[col] = df0[col].fillna(0)
            try:
                df_cart = get_bestsplit_list(df0, col)
                df_cart.append(min(df0[col]) - 100)
                df_cart.append(max(df0[col]) + 100)
                df_sort = sorted(df_cart)
                print(col, df_sort)
                colg = col + '_g'
                df0[colg] = pd.cut(df0[col], df_sort)
                df_woe = woe_iv2(df0, colg, 'target')
                # df_woe.drop(['bad_pcnt', 'good_pcnt'], axis=1, inplace=True)
                df_woe['sum_iv'] = sum(df_woe['IV'])
                df_woe['col'] = col
                print(df_woe)
                print(col, 'IV:', sum(df_woe['IV']))
                df_woe2 = df_woe[[colg, 'WOE']]
                df_woe2.rename(columns={'WOE': col + '_woe'}, inplace=True)
                df0 = pd.merge(df0, df_woe2, how='left', on=colg)
            except:
                continue
    return df0
