#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 11:38
# @Author  : AndrewMa
# @File    : scorecard.py

import pandas as pd
import statsmodels.api as sm


def LogisticRegressionModel(train_set, target, features):
    """
    Logistic Regression Model Via 'statsmodels' api instance

    :param train_set: 模型训练数据集
    :param target: 目标变量
    :param features: 进入训练的模型特征
    :return:
    """

    assert set(features) < set(train_set.columns), "Not all vars of feature list in data set!"

    y = train_set[target]
    X = train_set[features]
    X['intercept'] = 1.0

    LR = sm.Logit(y, X).fit()


def StepwiseSelectionFeatures(X, y, initial_list=[], threshold_in=0.01, threshold_out=0.05, verbose=True):
    """
    使用 Forward-Backward 方法进行特征选择，按照拟合 Logit 的 p-value 进行

    :param X: pandas.DataFrame 包含候选特征
    :param y: 拟合目标变量
    :param initial_list: 初始特征列表（包含在X.columns中）
    :param threshold_in: include a feature if its p-value < threshold_in
    :param threshold_out: exclude a feature if its p-value > threshold_out
    :param verbose: whether to print the sequence of inclusions and exclusions
    :return: 最终选定的特征列表
    """

    included = list(initial_list)
    while True:
        changed = False
        # Forward Step
        excluded =list(set(X.columns) - set(included))
        new_pval = pd.Series(index=excluded)
        for new_col in excluded:
            model = sm.Logit(y, sm.add_constant(X[included + [new_col]])).fit()
            new_pval[new_col] = model.pvalues[new_col]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed = True
            if verbose:
                print('Add {:30} with p-value {:.6}'.format(best_feature, best_pval))

        # Backward Step
        model = sm.Logit(y, sm.add_constant(X[included])).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            changed = True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
            if verbose:
                print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

        if not changed:
            break
    return included
