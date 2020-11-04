#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/11/4 20:42
# @Author  : AndrewMa
# @Email   : masculn@gmail.com
# @File    : model_utils.py

from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use('ggplot')


def compute_ks(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    ks = abs(tpr - fpr).max()
    return ks, fpr, tpr


def ks_score(clf, xx, yy):
    y_pred = clf.predict_proba(xx)[:, 1]  # 模型预测, 被分为坏样本的概率
    ks, _, _ = compute_ks(yy, y_pred)

    return ks


def model_eval(clf, xx, yy, show=False):
    """
    对训练好的模型，在数据集 [xx, yy] 上进行效果评估

    :param clf: 训练好的模型
    :param xx: 经过预处理的特征
    :param yy: 数据集实际 Y 值
    :param show: 是否打印评估值，默认否
    :return:
    """

    y_pred = clf.predict_proba(xx)[:, 1]  # 模型预测, 被分为坏样本的概率
    ks, fpr, tpr = compute_ks(yy, y_pred)
    auc = metrics.roc_auc_score(yy, y_pred)

    if show:
        print("ks: ", ks)
        print("auc: ", auc)

    return ks, auc


def imb_eval(clf, xx, yy, show=False):
    """
    专门针对不均衡样本分类问题，输出 PRCAUC指标（PRC曲线下的面积）
    :param clf:
    :param xx:
    :param yy:
    :param show:
    :return:
    """
    y_pred = clf.predict_proba(xx)[:, 1]  # 模型预测, 被分为坏样本的概率
    precision, recall, _ = metrics.precision_recall_curve(yy, y_pred)
    prc_auc = metrics.auc(recall, precision)
    ks, _, _ = compute_ks(yy, y_pred)

    if show:
        print("ks: ", ks)
        print("prc_auc: ", prc_auc)

    return ks, prc_auc


def model_lift(clf, xx, yy, nbin=10, show=False):
    """
    计算模型的 Lift 数据, 打分分组之后的识别能力

    :param clf:
    :param xx:
    :param yy:
    :param nbin:
    :param show:
    :return:
    """
    import pandas as pd

    y_pred = clf.predict_proba(xx)[:, 1]
    df = pd.DataFrame({'target': yy, 'pred': y_pred})
    df.sort_values(by='pred', ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['qbin'] = pd.qcut(df['pred'], nbin, labels=False)
    df['qbin'] = df['qbin'].apply(lambda x: nbin - x)

    bad = df.groupby('qbin')['target'].sum()
    total = df.groupby('qbin')['target'].count()
    lift_df = pd.DataFrame({"total": total, "bad": bad})
    lift_df['bad_rate'] = lift_df.apply(lambda x: x.bad / x.total, axis=1)
    lift_df['bad_pcnt'] = lift_df['bad'].sum() / lift_df['total'].sum()
    lift_df['lift'] = lift_df.apply(lambda x: x.bad_rate / x.bad_pcnt, axis=1)

    if show:
        plt.figure(figsize=(10, 6))
        plt.plot(lift_df.index, lift_df.lift, 'r--')
        plt.plot(lift_df.index, [1.0] * nbin, 'b-')
        plt.xticks(range(1, nbin + 1))
        plt.xlabel("bin")
        plt.ylabel("lift: bad_rate / bad_pcnt")
        plt.title("Model Forecast Lift Chart")
        plt.legend(labels=['lift', 'avg'], loc='best')
        plt.show()


def plot_ks(y_true, y_pred):
    ks, fpr, tpr = compute_ks(y_true, y_pred)
    idx = list(tpr - fpr).index(ks)

    n_sample = len(tpr)
    x_axis = [float(i) / n_sample for i in range(n_sample)]

    plt.figure()
    plt.plot(x_axis, tpr, 'r')
    plt.plot(x_axis, fpr, 'b')
    plt.plot([idx / n_sample, idx / n_sample], [0, 1], 'g--',
             label=('quantile: ' + str(round(idx / n_sample, 3))))
    plt.xlabel('Quantile of model score (high -> low)')
    plt.ylabel('Cumulative capture rate')
    plt.title('KS value (' + str(round(ks, 3)) + ')')
    plt.grid(True)
    plt.show()


def plot_roc(clf, xx, yy, label='test'):
    """
    绘制分类器预测数据的 ROC 曲线 （Recevier Operation Curve）
    :param clf:
    :param xx:
    :param yy:
    :param label:
    :return:
    """
    y_pred = clf.predict_proba(xx)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(yy, y_pred)

    plt.figure()
    plt.plot(fpr, tpr, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def plot_prc(clf, xx, yy, label='test'):
    """
    绘制分类器预测数据的 PRC 曲线 （Precision Recall Curve）
    :param clf:
    :param xx:
    :param yy:
    :param label:
    :return:
    """
    y_pred = clf.predict_proba(xx)[:, 1]
    precision, recall, _ = metrics.precision_recall_curve(yy, y_pred)

    plt.figure()
    plt.plot(recall, precision, label=label)
    # plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc='best')
    plt.show()
