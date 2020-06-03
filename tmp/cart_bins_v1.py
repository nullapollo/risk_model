import sys
import pandas as pd
import numpy as np
import time

def calc_score_median(sample_set, var):

    var_list = list(np.unique(sample_set[var]))
    var_median_list = []
    for i in range(len(var_list) - 1):
        var_median = (var_list[i] + var_list[i + 1]) / 2
        var_median_list.append(var_median)
    return var_median_list


def choose_best_split(sample_set, var, min_sample,str_y):

    score_median_list = calc_score_median(sample_set, var)
    median_len = len(score_median_list)
    sample_cnt = sample_set.shape[0]
    sample1_cnt = sum(sample_set[str_y])
    sample0_cnt = sample_cnt - sample1_cnt
    Gini = 1 - np.square(sample1_cnt / sample_cnt) - np.square(sample0_cnt / sample_cnt)

    bestGini = 0.0
    bestSplit_point = 0.0
    bestSplit_position = 0.0
    for i in range(median_len):
        left = sample_set[sample_set[var] < score_median_list[i]]
        right = sample_set[sample_set[var] > score_median_list[i]]

        left_cnt = left.shape[0]
        right_cnt = right.shape[0]
        left1_cnt = sum(left[str_y])
        right1_cnt = sum(right[str_y])
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


def bining_data_split(sample_set, var, min_sample, split_list,str_y):

    split, position = choose_best_split(sample_set, var, min_sample,str_y)
    if split != 0.0:
        split_list.append(split)
    sample_set_left = sample_set[sample_set[var] < split]
    sample_set_right = sample_set[sample_set[var] > split]
    if len(sample_set_left) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_left, var, min_sample, split_list,str_y)
    else:
        None
    if len(sample_set_right) >= min_sample * 2 and position not in [0.0, 1.0]:
        bining_data_split(sample_set_right, var, min_sample, split_list,str_y)
    else:
        None

def get_bestsplit_list(sample_set, var,str_y,min_rate=0.05):
    min_df = sample_set.shape[0] * min_rate
    split_list = []
    bining_data_split(sample_set, var, min_df, split_list,str_y)
    return split_list