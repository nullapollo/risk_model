
import pandas as pd
import numpy as np

from local.woe import get_bestsplit_list,woe_iv2
'''
分箱，空值划分为单独的一类
包括数据dataframe不要设空值为默认值
'''
def get_woe_data_split_null(df0,df,cols):
    for col in cols:
        if df0[col].dtype == 'object':
            df0[col]=df0[col].fillna('0')
            df0[col]=df0[col].replace({None :'0'})
            df_woe = woe_iv2(df0, col, 'target')
            df_woe['sum_iv']=sum(df_woe['IV'])
            df_woe['col']=col
            df_woe['bad_rate'] = df_woe['bad']/df_woe['total']
            df_woe['good_rate'] = df_woe['good']/df_woe['total']
            print(df_woe)
            print(col, 'IV:', sum(df_woe['IV']))
            df=pd.concat([df, df_woe], axis=0)
        elif len(df0[col].unique())<5:
            print(col,df0[col].unique())
            df0[col]=df0[col].fillna(0)
            df_woe = woe_iv2(df0, col, 'target')
            df_woe['sum_iv']=sum(df_woe['IV'])
            df_woe['col']=col
            df_woe['bad_rate'] = df_woe['bad']/df_woe['total']
            df_woe['good_rate'] = df_woe['good']/df_woe['total']
            print(df_woe)
            print(col, 'IV:', sum(df_woe['IV']))
            df=pd.concat([df, df_woe], axis=0)
        else:
            # 新增两列
            col_str=col+'_str'
            col_g = col + '_g'
            
            # 赋值空值
            df0[col_g]='null'
            
            # 取出 空值 和 非空值
            df0[col_str]=df0[col].apply(lambda x:str(x))
            df_not_null =df0[(df0[col_str]!='nan') & (df0[col_str]!='')]
            df_null=df0[(df0[col_str]=='nan') | (df0[col_str]=='')]
            

            # 计算 woe值
            df_woe_not_null = None
            if df_not_null.empty==False:
                # 非空值 分箱
                df_cart = get_bestsplit_list(df_not_null, col)
                df_cart.append(min(df_not_null[col])-0.001)
                df_cart.append(max(df_not_null[col]))
                df_sort = sorted(df_cart)
                df_not_null[col_g] = pd.cut(df_not_null[col], df_sort).tolist()
                df_woe_not_null = woe_iv2(df_not_null, col_g, 'target')
            
            df_woe_null = None
            if df_null.empty==False:
                df_woe_null = woe_iv2(df_null, col_g, 'target')
            
            df_woe = df_woe_null if df_woe_not_null is None else df_woe_not_null
            if df_woe_null is not  None and df_woe_not_null is not None:
                df_woe = pd.concat([df_woe_null,df_woe_not_null])
                df_woe.reset_index(inplace=True)
            
            # 计算 woe 与 iv
            df_woe['sum_iv']=sum(df_woe['IV'])
            df_woe['col']=col
            df_woe['bad_rate'] = df_woe['bad']/df_woe['total']
            df_woe['good_rate'] = df_woe['good']/df_woe['total']

            print(col,df_sort)
            print(df_woe)
            print(col, 'IV:', sum(df_woe['IV']))
            
            # 删除新增列
            df_not_null.drop(col_g, axis=1, inplace=True)
            df_null.drop(col_g, axis=1, inplace=True)
            df0.drop(col_g, axis=1, inplace=True)
            df0.drop(col_str, axis=1, inplace=True)
            
            df=pd.concat([df,df_woe], axis=0)
    return df

'''
获取列分箱结果及woe值
返回是否是连续型变量标识
连续性变量，返回分隔点 及 对应woe值
非连续性变量，返回类型列表 及 对应woe值
'''
def get_woe_bin_split_null(df,col,target):
    obj_reuslt ={}
    
    # 分类类型 或者 唯一数小于5
    if df[col].dtype == 'object' or len(df[col].unique())<5:
        if df[col].dtype == 'object':
            df[col]=df[col].fillna('0')
            df[col]=df[col].replace({None :'0'})
        else:
            df[col]=df[col].fillna(0)
            #df[col]=df[col].replace({None :0})
        # 计算 woe 与 bin
        df_woe = get_woe_by_group(df,col,target)
        arr_woe = []
        arr_bins = []
        for v in df_woe['WOE']:
            arr_woe.append(v)
        for v in df_woe['group']:
            arr_bins.append(v)
            
        obj_reuslt['is_continue']=False 
        obj_reuslt['woe']=arr_woe 
        obj_reuslt['bin']=arr_bins 
        
    else:
        df.fillna(0,inplace=True)
        # 分箱，获取分箱结果
        df_cart = get_bestsplit_list(df, col)
        # 插入最大值，最小值
        df_cart.append(min(df[col])-0.001)
        df_cart.append(max(df[col]))
        df_sort = sorted(df_cart)
        #print(col,df_sort)
        # 根据分箱分组
        col_g = col + '_g'
        df[col_g] = pd.cut(df[col], df_sort)
        # 计算分组的woe值
        df_woe = get_woe_by_group(df,col_g,target)
        arr_woe = []
        arr_bins = df_sort
        for v in df_woe['WOE']:
            arr_woe.append(v)
        obj_reuslt['is_continue']=True 
        obj_reuslt['woe']=arr_woe 
        obj_reuslt['bin']=arr_bins

        df.drop(col_g, axis=1, inplace=True)
    
    return obj_reuslt
    
'''
根据已分组dataframe计算woe值
'''
def get_woe_by_group(df, col, target):
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
    regroup['bad_pcnt'].replace(0,0.000001,inplace=True)
    regroup['good_pcnt'].replace(0,0.000001,inplace=True)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/x.bad_pcnt),axis = 1)
    regroup.rename(columns={col:'group'},inplace=True)
    
    return regroup

'''
根据 get_woe_bin方法得到参数，返回列数据的woe值
'''
def get_woe_data_by_binwoedata_split_null(df,col,obj_bin_woe):
    arr_woe_data = []
    arr_bin = obj_bin_woe['bin']
    arr_woe = obj_bin_woe['woe']
    if obj_bin_woe['is_continue']:
        len_bin = len(arr_bin)
        for v in df[col]:
            for i in range(len_bin-1):
                if v>arr_bin[i] and v<=arr_bin[i+1]:
                    arr_woe_data.append(arr_woe[i])
                    break;
    else:
        for v in df[col]:
            arr_woe_data.append(arr_woe[arr_bin.index(v)])
    
    return arr_woe_data
        
