#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/4 20:38
# @Author  : AndrewMa
# @Email   : masculn@gmail.com
# @File    : hyper.py

# 自动化参数搜索，使用 hyperopt 配合 xgb 模型训练
# 思路：
# 目标函数： objective(params) --> metric (scalar, user defined)
# 设定1： train set 跑 CV, 找 test 上 auc 平均最大的参数 --> max mean(test_auc), 当然也可以看 ks, max mean(test_ks)
# 设定2： train set 跑 fit，找 oot 上 ks 足够大（预测能力好），且 train 和 oot 上 ks 足够接近（稳定性）
#        --> max oot_ks + weight * diff(train_ks, oot_ks)
# 参数空间： params space, xgb 的运行参数
# 目标函数 + 参数空间 喂给 hyperopt 的优化函数，找极值点

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import hyperopt

from rkmd.utils.model_eval import ks_score


class HyperXgb(object):
    def __init__(self, train, oot, feature, target):
        """

        :param train:
        :param oot:
        :param feature:
        :param target:
        """
        # 定义 data set & feature list
        self.train_x = train[feature]
        self.train_y = train[target]
        self.oot_x = oot[feature]
        self.oot_y = oot[target]
        self.target = target
        self.feature = feature

        # 定义 xgboost clf
        self.xgb = xgb.XGBClassifier(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=2,
            min_child_weight=1,
            gamma=1.,
            subsample=1.,
            colsample_bytree=1.,
            reg_lambda=0,
            reg_alpha=1,
            nthread=-1,
            random_state=123,
            n_jobs=-1
        )

    def cv_eval(self, params, cv_split=10, cv_rpt=3, cv_metric='auc'):
        """
        在 train 上使用 cv 方法训练模型，评估 test 上 metric 的均值

        :param params: 需要调整的 hyperparameter dict
        :param cv_split: cv 的 KFold 个数
        :param cv_rpt: cv 的 重复 KFold 次数
        :param cv_metric: test set 上效果的度量, 支持 'auc' & 'ks'
        :return:
        """
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth'])
        self.xgb.set_params(**params)

        if cv_metric == 'auc':
            score_metric = 'roc_auc'
        elif cv_metric == 'ks':
            score_metric = ks_score  # user defined model metric
        else:
            raise ValueError("Invalid Cross Validation Metric: %s" % cv_metric)

        cv = RepeatedStratifiedKFold(n_splits=cv_split, n_repeats=cv_rpt, random_state=100)
        score = cross_val_score(self.xgb, self.train_x, self.train_y, scoring=score_metric, cv=cv, n_jobs=-1)

        return -1.0 * np.mean(score)  # 这里注意下，cv 会返回所有的 case 结果列表，要对列表做处理

    def bus_eval(self, params, metric='ks', weight=0.2):
        """
        在 train 上训练模型，评估 train 和 oot 上的表现 ks 值， 加权 oot 上的绝对表现 和 train/oot 上表现的接近程度

        :param params: clf 的一组参数
        :param metric: 评估指标，默认 'ks' 可选 'auc'
        :param weight: 评估加权系数 weight >= 0, weight = 0 退化为 purely 在 oot 上评价
        :return:
        """
        if 'max_depth' in params.keys():
            params['max_depth'] = int(params['max_depth'])
        self.xgb.set_params(**params)

        self.xgb.fit(self.train_x, self.train_y)

        if metric == 'ks':
            metric_func = ks_score
        elif metric == 'auc':
            metric_func = roc_auc_score
        else:
            raise ValueError("Invalid Model Metric: %s" % metric)

        train_metric = metric_func(self.xgb, self.train_x, self.train_y)
        oot_metric = metric_func(self.xgb, self.oot_x, self.oot_y)

        return -1.0 * (oot_metric - abs(train_metric - oot_metric) * weight)

    def hyper_param(self, param_space, max_evals=500, obj_type='cv'):
        """
        在超参数空间，寻找 objective 的最优值，返回

        :param param_space:
        :param max_evals:
        :param obj_type: 需要优化的目标函数类型
        :return:
        """
        if obj_type == 'cv':
            obj_func = self.cv_eval
        elif obj_type == 'bus':
            obj_func = self.bus_eval
        else:
            raise ValueError("Invalid Objective Type: %s" % obj_type)

        best = hyperopt.fmin(obj_func, param_space, algo=hyperopt.tpe.suggest, max_evals=max_evals)
        print(best)

        return best


if __name__ == "__main__":
    from rkmd.dts import load_ds

    train_df, test_df, oot_df, feature_x, ex_list = load_ds.load_sc()

    model_df = pd.concat([train_df, test_df])
    model_df.reset_index(drop=True, inplace=True)

    hx = HyperXgb(train=model_df, oot=oot_df, feature=feature_x, target='target')

    # 自定义要调的超参数，搜索空间，以及需要优化的目标函数
    param_dist = {
        'max_depth': hyperopt.hp.quniform('max_depth', 2, 14, 1),
        # 注意不要使用hyperopt.hp.choice,因为他返回最优点在list中的下标,很容易造成误解,深坑一个!!!!!!!
        'min_child_weight': hyperopt.hp.quniform('min_child_weight', 1, 100, 1),
        'subsample': hyperopt.hp.quniform('subsample', 0.5, 1, 0.05),
        'gamma': hyperopt.hp.quniform('gamma', 0.5, 10, 0.05),
        'colsample_bytree': hyperopt.hp.quniform('colsample_bytree', 0.5, 1, 0.05)
    }
    best_params = hx.hyper_param(param_space=param_dist, max_evals=50, obj_type='bus')
