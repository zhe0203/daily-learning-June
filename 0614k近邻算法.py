import numpy as np
import operator

def createDataSet():
	group = array([1,1.1],[1,1],[0,0],[0,0.1])
	labels = ['A','A','B','B']
	return(group,labels)

# k-近邻算法
def classify0(inX,dataSet,labels,k):      # inx用于分类的目标数据，dataset为现有数据，labels为现有数据的标签，k为选择的近邻数目
	dataSetSize = dataSet.shape[0]
	diffMat = tile(inX,(dataSetSize,1))-dataSet    # 将inX数组使用tile函数重复dataset行数一行的数据以便计算
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()       # 对于距离进行排序
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0)+1        # 如果键存在，返回对应的值，否则返回0
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # itemgetter 获取对象的第1个域的值
	return(sortedClassCount[0][0])

def file2matrix(filename):
	fr = open(filename)
	arrayolines = fr.readlines()
	numberoflines = len(arrayolines)   # 得到文件行数
	returnmat = np.zeros((numberoflines,3))   # 创建返回的numpy矩阵
	classlabelvector = []
	index = 0
	for line in arrayolines:
		line = line.strip()    # 去除空格
		listfromline = line.split('\t')
		returnmat[index,:] = listfromline[0:3]
		classlabelvector.append(int(listfromline[-1]))
		index += 1
	return(returnmat,classlabelvector)

import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(dataingmat[:,1],dataingmat[:,2],15*array(datinglabels),15*array(datinglabels))
plt.show()

# 数据归一化处理
def autonorm(dataset):
	minvals = dataset.min(axis=0)   # 对于数据的列取最小值
	maxvals = dataset.max(axis=0)
	ranges = maxvals - minvals
	normdataset = np.zeros(shape(dataset))
	m = dataset.shape(0)
	normdataset = dataset -np.tile(minval,(m,1))          # 这里使用python的广播性质，也可以使用dataset-minval
	normdataset = normdataset/np.tile(ranges,(m,1))       # 广播性质，(data-minval)/(maxval-minval)
	return(normdataset,ranges,minvals)

# 评估计算的代码
def datingclasstest():
	horario = 0.1
	datingdatamat,datinglabels = file2matrix('datingtest.txt')
	normmat,ranges,minvals = autonorm(datingdatamat)   # 对于数据进行标准化处理
	m = normmat.shape(0)    # 返回标准化后数据的行数
	numtestvecs = int(m*horario)   # 生成测试数据行数
	errorcount = 0
	for i in range(numtestvecs):
		classresult = classify0(normmat[i,:],normmat[numtestvecs:m,:],datinglabels[numtestvecs:m],3)  # 前m行作为测试数据v
		if classresult != datinglabels[i]:
			errorcount += 1
	print('the total error rate is: %f' % (errorcount/float(numtestvecs)))

# 对未知数据进行预测
def classifyperson():
	resultlist = ['not at all','in small doses','in large doses']
	percenttats = float(input('percentage of time'))   # 输入数据
	ffmiles = float(input('liters of ice cream'))      # 输入数据
	iceream = float(input('liter of ice cream'))       # 输入数据
	normmat,ranges,minvals = autonorm(datingdatamat)   # 返回数据标准化结果
	inarr = np.array([ffmiles,percenttats,iceream])    # 将输入数据组合成数组形式
	classifierresult = classify0((inarr-minvals)/ranges,normmat,datinglabels,3)
	print('you will probably like this person:',resultlist[classifierresult - 1])

# 将图像转换成测试向量
def img2vector(filename):
	returnvect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linestr = fr.readline()
		for j in range(32):
			returnvect[0,32*i+j] = int(linestr[j])
	return(returnvect)

fr = open(r'C:\Users\jk\Desktop\命令提示符.txt','r')
data = []
for i in fr.readlines():
    line = i.strip()
    data.append(int(line))                    # 如果使用append
print(data)

# 测试算法：使用k-近邻算法识别手写数字
from os import listdir
def handwritingclasstest():
	hwlabels = []
	trainingfilelist = lisdir('trainingdigits')   # 获取目录内容
	m = len(trainingfilelist)　　　# 计算存在内容的数据
	trainingmat = np.zeros((m,1024))
	for i in range(m):      # 对于每一幅图像作如下的处理
		filenamestr = trainingfilelist[i]
		filestr = filenamestr.split('.')[0]
		classnumstr = int(filestr.split('_')[0])  # 9_45.txt 9的45例
		hwlabels.append(classnumstr)                    # 将分类标签形成单独的文件夹
		trainingmat[i,:] = img2vector('trainingdigits\%s' % (filenamestr))   # 载入文件，将其转换为数字
	testfilelist = listdir('testdigits')   # 载入测试数据的文件夹
	errorcount = 0
	mtest = len(testfilelist)
	for i in range(mtest):
		filenamestr = testfilelist[i]
		filestr = filenamestr.split('.')[0]
		classnumstr = filenamestr.split('_')[0]
		vectorundertest = img2vector('testfilelist\%s' % (filenamestr))
		classifierresult = classify0(vectorundertest,trainingmat,hwlabels,3)
		if classifierresult != classnumstr:
			errorcount += 1
	print('the total error rate is: %f' % (errorcount/float(numtestvecs)))
