# http://www.cnblogs.com/kui2/archive/2013/03/15/2961778.html
import sys
import numpy as np

def Avg(data,scoreMap):
    x1 = np.array(data)
    x2 = np.array(scoreMap)
    totalScore = np.dot(x1,x2)     # 数据相乘运算   与 x1 * x2是有区别的
    # totalScore = float(sum(map(lambda t:t[0]*t[1], map(data,scoreMap))))
    totalN = sum(data)
    return totalScore / totalN

def Wilson(p, n, maxScore):
    # p = float(p)
    K = 1.96 # 95% confidence level
    _K2_div_n = (K ** 2) / n
    pmin = (p + _K2_div_n / 2.0 - K*((p*(1-p)/n + _K2_div_n/n/4.0)**0.5)) / (1 + _K2_div_n)
    return pmin * maxScore

def WilsonAvgP(n):
    totalP = 0.0; totalN = 0
    p = 0.01
    while True:
        totalP += Wilson(p, n, 1); totalN += 1
        p += 0.01
        if p >= 1: break
    return totalP / totalN

def Bayesian(C, M, n, s):
    return (C*M + n*s) / (n + C)

DATA = {
    "A" : (2,0,0,0,0),
    "B" : (1,0,3,8,40),
    "C" : (4,2,1,5,10),
    # "D" : (500,250,100,250,500),
    # "E" : (0,0,1,0,0),
    # "F" : (10,0,5,0,10),
    # "G" : (10,10,10,10,10),
    # "H" : (100,50,20,50,100),
    }
SCORE_MAP = (1.0,2.0,3.0,4.0,5.0)
MAX_SCORE = 5

result = {} # key : avgScore, wilsonScore, wilsonRank
C = 64; M = WilsonAvgP(C) * MAX_SCORE
for k,v in DATA.items():
    n = sum(v)
    avgScore = Avg(v,SCORE_MAP)
    wilsonScore = Wilson(avgScore / MAX_SCORE, n, MAX_SCORE)
    wilsonRank = Bayesian(C, M, n, wilsonScore)

    result[k] = (avgScore, wilsonScore, wilsonRank)
for k in sorted(result.keys()):
    v = result[k]
    # print(v)
    print(k,":","%.2f" % v[0]," ","%.2f" % v[1]," ","%.2f" % v[2])
