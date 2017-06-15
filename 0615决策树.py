# 计算给定数据集的熵  信息熵
from math import log
def calcshanonent(dataset):
    numentries = len(dataset)
    labelcounts = {}
    for featvec in dataset:
        currentlabel = featvec[-1]
        if currentlabel not in labelcounts.keys():    # 为TRUE则执行，否则不执行
            labelcounts[currentlabel] = 0
        labelcounts[currentlabel] += 1                # if 为False 则执行这一句
    shannonent = 0
    for key in labelcounts:
        prob = float(labelcounts[key])/numentries
        shannonent -= prob * log(prob,2)
    return(shannonent)

# 按照给定特征划分数据集
def splitdataset(dataset,axis,value):
    retdataset = []
    for featvec in dataset:
        if featvec[axis] == value:
            reducefeatvec = featvec[:axis]
            reducefeatvec.extend(featvec[axis+1:])
            retdataset.append(reducefeatvec)
    return(retdataset)

# 计算信息增益
def choosebestfeaturetosplit(dataset):
    numfeatures = len(dataset[0]) - 1     # 判断数据集当前包含的特征数
    baseentropy = calcshanonent(dataset)   # 计算信息熵
    bestinfogain = 0
    bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in dataset]   # 将第i个特征的所有值写入这个list中
        uniquevals = set(featlist)   # 创建唯一的分类标签
        newentropy = 0
        for value in uniquevals:
            subdataset = splitdataset(dataset,i,value)
            prob = len(dataset)/float(len(dataset))
            newentropy += probs * calcshanonent(subdataset)
        infogain = baseentropy - newentropy
        if infogain > bestinfogain:
            bestinfogain = infogain
            bestfeature = i
    return(bestfeature)

#
def majoritycnt(classlist):
    classcount = {}
    for vote in classcount:
        if vote not in classcount.keys():
            classcount[vote] = 0
        sortedclasscount = sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
        return(sortedclasscount[0][0])

# 创建数的函数代码
def createtree(dataset,labels):
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):  # 类别完全相同时则停止继续划分
        return(classlist[0])
    if len(dataset[0]) == 1:
        return(majoritycnt(classlist))
    bestfeat = choosebestfeaturetosplit(dataset)
    bestfeatlabel = labels[bestfeat]
    mytree = {bestfeatlabel:{}}
    del (labels[bestfeat])
    featvalues = [example[bestfeat] for example in dataset]
    uniquevals = set(featvalues)
    for values in uniquevals:
        sublabels = labels[:]
        mytree[bestfeatlabel][value] = createtree(splitdatset(
            dataset,bestfeat,values),sublabels)
    return(mytree)
