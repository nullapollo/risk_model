#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/6/10 15:04
# @Author  : AndrewMa
# @File    : test.py

import pandas as pd
from crsc.bins.best_ks import best_ks_bin

data_all = pd.read_csv("E:/Models/IntelligentRiskControl/第6章/scorecard.txt")

dev_data = data_all.loc[data_all['samp_type'] == 'dev']
val_data = data_all.loc[data_all['samp_type'] == 'val']
off_data = data_all.loc[data_all['samp_type'] == 'off']

result, _ = best_ks_bin(flag_name='bad_ind', factor_name='act_info', data=dev_data)
result.reset_index(level=0, inplace=True)
result2, _ = best_ks_bin(flag_name='bad_ind', factor_name='act_info', data=dev_data, piece=3)
result2.reset_index(level=0, inplace=True)


def bin_plot(df, var):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('%s-bins' % var)
    ax1.set_ylabel('woe', color=color)
    ax1.bar(df['index'], df['WOE'], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('bad_rate', color=color)  # we already handled the x-label with ax1
    ax2.plot(df['index'], df['bad_rate'], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


bin_plot(result2, 'act_info')
