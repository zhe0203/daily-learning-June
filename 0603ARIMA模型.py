# -*- coding: utf-8 -*-
# 1、异常值处理——使用移动中位数的方法进行异常值的检验
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 构造含有异常的序列
df = np.random.randn(77)
df = list(df)
df.insert(5,7)
df.insert(16,-9)
df.insert(50,10)
df = pd.DataFrame(df,index=pd.date_range('20130101', periods=80),columns=['u'])
# 绘制含有异常值的图形
# plt.plot(mydata)
# plt.show()

# 定义get_median_filtered函数来进行异常值的判断
def get_median_filtered(signal,threshold=3):
    signal = signal.copy()
    difference = np.abs(signal - np.median(signal))    # 对于每一个数值减去其中位数
    median_difference = np.median(difference)          # 对于作差后的数据求其中位数
    if median_difference == 0:
        s = 0
    else:
        s = difference / float(median_difference)
    mask = s > threshold
    signal[mask] = np.median(signal)
    return signal

figsize = (7,2.75)
kw = dict(marker='o',linestyle='none',color='r',alpha=3)
df['u_medf'] = get_median_filtered(df['u'].values,threshold=3)
outlier_idx = np.where(df['u_medf'].values != df['u'].values)[0]
fig,ax = plt.subplots(figsize=figsize)
df['u'].plot()
df['u'][outlier_idx].plot(**kw)
plt.show()

# 改进方法  FFT方法
def detect_outlier_position_by_fft(signal,threshold_freq=0.1,frequency_amplitude=0.001):
    signal = signal.copy()
    fft_of_signal = np.fft.fft(signal)
    outlier = np.max(signal) if abs(np.max(signal)) > abs(np.min(signal)) else np.min(signal)
    if np.any(np.abs(fft_of_signal[threshold_freq:]) > frequency_amplitude):
        index_of_outlier = np.where(signal == outlier)
        return index_of_outlier[0]
    else:
        return None
outlier_idx = []
y = df['u'].values
opt = dict(threshold_freq=0.01,frequency_amplitude=.001)
win = 20
for k in range(win*20,y.size,win):
    idx = detect_outlier_position_by_fft(y[k-win:k+win],**opt)
    if idx is not None:
        outlier_idx.append(k+idx[0]-win)
outlier_idx = list(set(outlier_idx))
fig,ax = plt.subplots(figsize=figsize)
# df['u'].plot()
# df['u'][outlier_idx].plot(**kw)
# plt.show()

# 使用pandas中的移动中位数方法进行异常值检验
'''
主要思想：
对于连续的n个数值，得到其中位数，然后用真实值分别前去中位数在去绝对值，得到每一个差是否大于给定的阀值，大于的话为异常值
'''

from pandas import rolling_median
threshold = 3                  # 指的是判断一个点位异常值的阀值
df['pandas'] = rolling_median(df['u'],window=3,center=True).fillna(method='bfill').fillna(method='ffill')

difference = np.abs(df['u'] - df['pandas'])  #　df['u']为原始数据  df['pandas'] 是求移动中位数后的结果
outlier_idx = difference > threshold
fig,ax = plt.subplots(figsize=figsize)
df['u'].plot()
df['u'][outlier_idx].plot(**kw)
plt.show()

# 缺失值处理
'''
1、用序列的均值替代，这样的好处是在计算方法的时候不受影响，但是连续的几个nan即使这样代替也会在差分的时候重现变为nan
2、直接删除，需在nan不太多的情况下这样做
'''

# 平稳性检验
## 序列的平稳性检验是进行实践序列分析的前提条件，只要运用ADF检验
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    dftest = adfuller(timeseries,autolag='AIC')
    return dftest[1]   # 返回的是p值
print(test_stationarity(df['u']))

# 不平稳的处理
'''
1、对数处理
2、差分：差分几阶合理？在保证ADF检验的p<0.01的情况下，阶数越小越好。否则会带来样本减少，还原序列麻烦，预测困难的问题
'''
def best_diff(df,maxdiff=8):
    p_set = {}
    for i in range(0,maxdiff):
        temp = df.copy()   # 每次循环前，重置
        if i == 0:
            temp['diff'] = temp['u']   #　temp.columns[1]
        else:
            temp['diff'] = temp['u'].diff(i)
            temp = temp.drop(temp.iloc[:i].index)  # 差分后，前几行的数据会变成nan，需要删掉
        pvalue = test_stationarity(temp['diff'])
        p_set[i] = pvalue
        p_df = pd.DataFrame.from_dict(p_set,orient='index')
        p_df = pd.DataFrame(p_df)
        # p_df.columns = ['p_value']
    i = 0
    while i < len(p_df):
        if p_df.iloc[i,0] < 0.01:
            bestdiff = i
            break
        i += 1
    return bestdiff
print(best_diff(df))

# 随机性检验
## 只有时间序列不是一个白噪声，该序列才能做分析
from statsmodels.stats.diagnostic import acorr_ljungbox
def test_stochastic(ts):
    p_value = acorr_ljungbox(ts,lags=1)  # lags可自定义
    return p_value

# 确定ARMA阶数
'''
1、通过自相关、偏相关图进行得到
2、借助AIC BIC统计量自动确定
对于多个时间序列需要分别预测，所以要选取自动的方式，而BIC可以有效对模型的过拟合，因而选定BIC作为判断的标准
'''
from statsmodels.tsa.arima_model import ARMA
def proper_model(data_ts,maxLap):
    init_bic = float('inf')
    init_p = 0
    init_q = 0
    init_properModel = None
    for p in np.arange(maxLag):
        for q in np.arange(maxLag):
            model = ARMA(data_ts,order=(p,q))
            try:
                results_ARMA = model.fit(disp=-1,method='css')
            except:
                continue
            bic = results_ARMA.bic
            if bic < init_bic:
                init_p = p
                init_q = q
                init_properModel = results_ARMA
                init_bic = bic
    return init_bic,init_p,init_q,init_properModel

# 在statsmodels包里还有更直接的函数
import statsmodels.tsa.stattools as st
order = st.arma_order_select_ic(timeseries,max_ar=5,max_ma=5,ic=['aic','bic','hqic'])
order.bic_min_order      # timeseries是待输入的时间序列

# 拟合ARMA
model = ARMA(timeseries,order=order.bic_min_order)
result_arma = model.fit(siasp=-1,method='css')

'''
对于差分后的时间序列，运用ARMA时间序列模型被称为ARIMA
在代码层面改写为model = ARIMA(timeseries,order=(p,d,q))
但是实际上，用差分过的序列之间进行ARMA建模更为方便，滞后添加一步还原的操作即可
'''

# 预测的y值还原
## 由于放入模型进行拟合的数据是经过对数或差分处理的数据，因而拟合得到的预测
## y值要经过差分和对数的还原才可与原观测值比较
def predict_recover(ts):
    ts = np.exp(ts)
    return ts

# 判断拟合优度
# 在机器学习领域，回归常用RMSE（均方根误差）来判断模型拟合的效果，
# RMSE衡量是预测值与实际值的差距，R2是预测值与均值之间的差距
train_predict = result_arma.predict()
train_predict = predict_recover(train_predict)    # 还原数据
RMSE = np.sqrt(((train_predict-timeseries)**2).sum()/timeseries.size)
'''
注意：使用statsmodel包可能只能对于之后一天进行预测，多天无法预测
'''

# 预测的代码
for t in range(len(test)):
    model = ARIMA(history,order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test(t)
    history.append(obs)
    print('predicted=%f,expected=%f' % (yhat,obs))
