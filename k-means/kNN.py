from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

def classify0(inX, dataSet, labels, k):
    """
    inX: 用于分类的输入向量
    dataSet: 输入的训练样本
    labels: 标签向量集
    k: 最近邻居的数目
    """
    #to calculate the distance
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    #得到距离最小的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0)+1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0,0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

#verify result with classify0, input [0, 0] should get result 'B'
group, labels = createDataSet()
result = classify0([0,0], group, labels, 3)
print(result)

#将文本文件转换成numpy的matrix
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

datingMat, datingLabels = file2matrix("datingTestSet2.txt")
print(datingMat)
print(datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingMat[:,1], datingMat[:,2])
plt.show()
