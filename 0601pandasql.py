# -*- coding: utf-8 -*-
'''
差分变化可以消除数据对时间的依赖性，也就是降低时间对数据的影响，这些影响通常包括数据的变化趋势以及数据周期性变化规律
进行差分操作时，一般用现在的观测值减去上个时刻的值就得到差分结果
若进行一次差分之后，时间项的作用并没有完全去掉，将会继续对差分的结果进行差分变化，直至完全消除
'''
import pandas as pd
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
# 数据来源于三年来每月洗发水的销售情况
def parser(x):
    return datetime.strptime('190'+x,'%Y-%m')

series = read_csv(r'sales-of-shampoo.csv',header=0,parse_dates=[0],\
                  index_col=0,squeeze=True,date_parser=parser)
series.plot()
pyplot.show()

# 手动差分
'''
在这一部分中，我们将会定义一个函数来实现差分变换，这个函数将会对提供的数据进行遍历
并根据指定的时间间隔进行差分变换
'''
def difference(dataset,interval=1):
    diff = list()
    for i in range(interval,len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

series = read_csv(r'sales-of-shampoo.csv',header=0,parse_dates=[0],\
                    index_col=0,squeeze=True,date_parser=parser)
x = series.values
diff = difference(x,2)
pyplot.plot(diff)
pyplot.show()

# 自动差分
'''
pandas库提供了一个函数可以自动计算数据的差分，diff()
'''

diff = series.diff()
pyplot.plot(diff)
pyplot.show()


# pandasql库学习  sqldf for pandas
from pandasql import sqldf,load_meat,load_births
# sqldf.sqldf主要接收两个参数，1：sql语句  2：一个环境变量locals() globals()
pysqldf = lambda q:sqldf(q,globals())

# 查询
# Any pandas dataframes will be automatically detected by pandasql
meat = load_meat()         # 数据
births = load_births()     # 数据
result = pysqldf('select * from meat limit 10;')
print(result.head())

# 数据的连接与整合
q = """select  m.date,m.beef,b.births from meat m inner join births b on m.date = b.date;"""
joined = pysqldf(q)
print(joined.head())

q = "select strftime('%Y',date) as year,sum(beef) as beef_total from meat group by year;"
print(pysqldf(q).head())
