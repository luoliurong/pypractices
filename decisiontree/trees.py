from math import log
import operator
import matplotlib.pyplot as plt

#===========================================================================================
#Section1: 理解什么是信息的期望值
def createDataSet():
    """
    创建一个dataset, 用于表示海洋生物
    第一列表示是否可以浮出水面，第二列表示是否有脚蹼，第三列判断是否属于鱼类
    """
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calShannonEnt(dataSet):
    """
    计算dataSet中所有类别的香农熵。熵 - 信息的期望值。
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannoEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannoEnt -= prob*log(prob,2)
    return shannoEnt

myDat, labels = createDataSet()
shannoVal = calShannonEnt(myDat)
print(shannoVal)
#改变一个数据，观察香农熵的变化
myDat[0][-1] = 'maybe'
shannoVal = calShannonEnt(myDat)
print(shannoVal)

#===========================================================================================
#Section 2: 划分数据集
def splitDataSet(dataSet, axis, value):
    """
    按照给定的特征划分数据集
    """
    retDataSet = []
    for featVet in dataSet:
        if featVet[axis] == value:
            reducedFeatVec = featVet[:axis]
            reducedFeatVec.extend(featVet[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

subDat1 = splitDataSet(myDat, 0, 1)
print(subDat1)
subDat2 = splitDataSet(myDat, 0, 0)
print(subDat2)

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0])-1
    baseEntropy = calShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

bFeature = chooseBestFeatureToSplit(myDat)
print(bFeature)

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet, labels):
    """
    创建树
    """
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeat]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

tree = createTree(myDat, labels)
print(tree)

#===========================================================================================
#Section 3: 使用matplotlib绘制树形图

def getNumLeafs(myTree):
    """
    获取叶节点的数目
    """
    numLeafs = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=="dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:   numLeafs += 1
    return numLeafs

def getTreeDepth(myTree):
    """
    获取树的层数
    """
    maxDepth = 0
    firstStr = myTree.keys()[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:   thisDepth = 1
        if thisDepth > maxDepth:    maxDepth = thisDepth
    return maxDepth


def retrieveTree(i):
    """
    辅助函数，用于测试方便
    """
    listOfTrees = [
        {'no surfacing':{0:'no', 1:{'flippers':{0:'no', 1:'yes'}}}},
        {'no surfacing':{0:'no', 1:{'flippers':{0:{'head':{0:'no', 1:'yes'}}, 1:'no'}}}}
    ]
    return listOfTrees[i]


def plotMidText(cntrPt, parentPt, txtString):
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[0])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = myTree.keys()[0]
    cntrpt = (plotTree.xOff + (1.0+float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    plotMidText(cntrpt, parentPt, nodeTxt)
    plotNode(firstStr, cntrpt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], cntrpt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrpt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrpt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeText, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", xytext=centerPt, textcoords="axes fraction", va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot(intree):
    fig=plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(intree))
    plotTree.totalD = float(getTreeDepth(intree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(intree, (0.5, 1.0), '')
    plt.show()


mytree = retrieveTree(0)
print(mytree)
#createPlot(mytree)