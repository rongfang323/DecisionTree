import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy


def DecisionTreeLearning(queueNode, depth, tree=None, max_d=0, treeList=None):
    print(max_d)
    if tree is None:
        tree = {}
    if treeList is None:
        treeList = []
    if max_d == depth:
        return
    newQueue = {}
    for q in queueNode:
        root, trainingExamples = q, queueNode[q]
        if np.sum(trainingExamples[:, 0] == 1) == 0:
            tree[root] = (None, 0)
            continue
        elif np.sum(trainingExamples[:, 0] == 0) == 0:
            tree[root] = (None, 1)
            continue

        value, feature = bestSplitNode(trainingExamples)
        tree[root] = (feature, value)
        leftChildren = trainingExamples[trainingExamples[:, feature] == value]
        rightChildren = trainingExamples[trainingExamples[:, feature] != value]

        lChildrenfirstClass = np.sum(leftChildren[:, 0] == 1)
        lChildrenSecondClass = np.sum(leftChildren[:, 0] == 0)
        rChildrenfirstClass = np.sum(rightChildren[:, 0] == 1)
        rChildrenSecondClass = np.sum(rightChildren[:, 0] == 0)

        if lChildrenfirstClass != 0 and lChildrenSecondClass != 0:
            newQueue[root * 2 + 1] = leftChildren

        if lChildrenfirstClass >= lChildrenSecondClass:
            tree[root * 2 + 1] = (None, 1)
        else:
            tree[root * 2 + 1] = (None, 0)

        if rChildrenfirstClass != 0 and rChildrenSecondClass != 0:
            newQueue[root * 2 + 2] = rightChildren
        if rChildrenfirstClass >= rChildrenSecondClass:
            tree[root * 2 + 2] = (None, 1)
        else:
            tree[root * 2 + 2] = (None, 0)

    treeList.append(deepcopy(tree))
    DecisionTreeLearning(newQueue, depth,
                         tree=tree, max_d=max_d + 1, treeList=treeList)
    return treeList


def DecisionTreeLearningDFirst(root, trainingExamples, depth, tree=None, max_d=0):
    if tree is None:
        tree = {}
    if max_d > depth:
        firstClass = np.sum(trainingExamples[:, 0] == 1)
        secondClass = np.sum(trainingExamples[:, 0] != 1)
        if firstClass >= secondClass:
            tree[root] = (None, 1)
        else:
            tree[root] = (None, 0)
        return
    value, feature = bestSplitNode(trainingExamples)
    tree[root] = (feature, value)
    leftChildren = trainingExamples[trainingExamples[:, feature] < value]
    rightChildren = trainingExamples[trainingExamples[:, feature] >= value]

    if np.sum(leftChildren[:, 0] == 1) != 0 and np.sum(leftChildren[:, 0] != 1) != 0:
        DecisionTreeLearningDFirst(root * 2 + 1,
                                        leftChildren,
                                        depth,
                                        tree=tree,
                                        max_d=max_d + 1)
    else:
        if np.sum(leftChildren[:, 0] == 1) == 0:
            tree[root * 2 + 1] = (None, 0)
        else:
            tree[root * 2 + 1] = (None, 1)

    if np.sum(rightChildren[:, 0] == 1) != 0 and np.sum(rightChildren[:, 0] != 1) != 0:
        DecisionTreeLearningDFirst(root * 2 + 2,
                                        rightChildren,
                                        depth,
                                        tree=tree,
                                        max_d=max_d + 1)

        if np.sum(leftChildren[:, 0] == 1) == 0:
            tree[root * 2 + 2] = (None, 0)
        else:
            tree[root * 2 + 2] = (None, 1)
    return tree


def bestSplitNode(trainingExamples):
    bestBenefit = -1
    bestValue = None
    bestFeature = None

    s = trainingExamples.shape[0]
    u = calculateU(np.sum(trainingExamples == 1), np.sum(trainingExamples == 0))
    for x in range(1, trainingExamples.shape[1]):
        trainingExamples = trainingExamples[np.argsort(trainingExamples[:, x])]
        lastY = None
        for i, y in enumerate(trainingExamples):
            if lastY is None:
                lastY = y[0]
            if lastY != y[0]:
                lastY = y[0]
                value = y[x]
                cl = trainingExamples[0: i]
                pl = i / s
                uAL = calculateU(np.sum(cl[:, 0] == 1), np.sum(cl[:, 0] == 0))
                cr = trainingExamples[i:]
                uAR = calculateU(np.sum(cr[:, 0] == 1), np.sum(cr[:, 0] == 0))
                pr = (s - i) / s

                benefit = u - pl * uAL - pr * uAR
                if benefit > bestBenefit:
                    bestValue = value
                    bestFeature = x
                    bestBenefit = benefit
    return bestValue, bestFeature


def calculateU(pPlus, pMin):
    s = pPlus + pMin
    return 2. * (pPlus / s) * (pMin / s)


def predict(vdata, tree):
    correctSum = 0
    predictYList = np.zeros((vdata.shape[0], 1))
    for i, d in enumerate(vdata):
        root = 0
        while True:
            if tree[root][0] is not None:
                if d[tree[root][0]] == tree[root][1]:
                    root = root * 2 + 1
                else:
                    root = root * 2 + 2
            else:

                predictY = tree[root][1]
                predictYList[i, 0] = predictY
                break
        if predictY == d[0]:
            correctSum += 1
    accuracy = correctSum / vdata.shape[0]
    return accuracy, predictYList


def RandomForestLearning(trainingExamples, numberTrees, numberFeatures, depth):
    forest = []
    for x in range(numberTrees):
        sampleFeatureList, sampleRoot = getsamples(trainingExamples, numberFeatures)
        Tree = DecisionTreeLearning({0: sampleRoot}, depth)[depth - 1]
        trueTree = {}
        for k, v in Tree.items():

            if v[0] is not None:
                trueTree[k] = (sampleFeatureList[v[0]], v[1])
            else:
                trueTree[k] = (None, v[1])
        forest.append(trueTree)
    return forest


def getsamples(trainingExamples, numberFeatures):
    totalFeature = range(1, trainingExamples.shape[1])

    featureList = np.random.choice(totalFeature, numberFeatures, replace=False)
    featureList = np.insert(featureList, 0, 0, axis=0)
    sampleList = np.random.choice(trainingExamples.shape[0], trainingExamples.shape[0], replace=True)
    sampleTreeRoot = np.zeros((sampleList.shape[0], featureList.shape[0]))
    for i, x in enumerate(sampleList):
        sampleTreeRoot[i, :] = trainingExamples[x, featureList]
    return featureList, sampleTreeRoot


def majVote(forest, data):
    allResult = np.zeros(shape=(data.shape[0], len(forest)))
    for i, x in enumerate(forest):
        accuracy, result = predict(data, x)
        result = np.reshape(result, (result.shape[0], 1))
        allResult[:, i] = result[:, 0]
    majorResult = np.sum(allResult, axis=1)
    majorResult[majorResult == 1] = 1
    majorResult[majorResult != 1] = 0
    accuracy = np.sum(majorResult == data[:, 0]) / data.shape[0]
    return accuracy

def readData(fileName):
    return np.genfromtxt(fileName, delimiter=',')

def draw(x, y, title, xlabel, ylabel, legend, fig):
    plt.figure(num=fig, figsize=(8, 5), )
    plt.plot(x, y, linewidth=1.0)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([legend])
    plt.show()


def drawMultiple(x, y1, y2, title, xlabel, ylabel, legend1, legend2, fig):
    plt.figure(num=fig, figsize=(8, 5), )
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend([legend1, legend2])
    plt.show()


if __name__ == '__main__':
    trainingData = readData('pa3_train.csv')
    validationData = readData('pa3_val.csv')
    testData = readData('pa3_test.csv')
    '''
    trainingData = np.delete(trainingData, 0, axis=0)
    validationData = np.delete(validationData, 0, axis=0)
    testData = np.delete(testData, 0, axis=0)
    '''
    depths = [1, 2, 3, 4, 5, 6, 7, 8]
    for depth in depths:
        treeList = DecisionTreeLearning({0: trainingData}, depth)
        validationAccuracyList = []
        trainingAccuracyList = []
        testAccuracyList = []
        for tree in treeList:
            a, b = predict(validationData, tree)
            a2, b2 = predict(trainingData, tree)
            a3, b3 = predict(testData, tree)
            validationAccuracyList.append(a)
            trainingAccuracyList.append(a2)
            testAccuracyList.append(a3)
        print(validationAccuracyList)
        print(trainingAccuracyList)
        print(testAccuracyList)

        draw(list(range(1, depth + 1)), validationAccuracyList, 'accuracy of valid data', 'depth', 'accuracy', 'valid', 1)
        draw(list(range(1, depth + 1)), trainingAccuracyList, 'accuracy of train data', 'depth', 'accuracy', 'train', 2)
        draw(list(range(1, depth + 1)), testAccuracyList, 'accuracy of test data', 'depth', 'accuracy', 'test', 2)

    n = [1, 2, 5, 10, 25]
    testAccuracyList2 = []
    validationAccuracyList2 = []
    trainingAccuracyList2 = []
    for i in n:
        forest = RandomForestLearning(trainingData, i, 5, 2)
        validationAccuracyList2.append(majVote(forest, validationData))
        trainingAccuracyList2.append(majVote(forest, trainingData))

    print(validationAccuracyList2)
    print(trainingAccuracyList2)
    draw(n, validationAccuracyList2, 'accuracy of valid data vs the number of trees', 'number of trees', 'accuracy', 'valid', 3)
    draw(n, trainingAccuracyList2, 'accuracy of train data vs the number of trees', 'number of trees', 'accuracy', 'train', 4)

    m = [1, 2, 5, 10, 25, 50]
    validationAccuracyList3 = []
    trainingAccuracyList3 = []
    
    for i in m:
        forest = RandomForestLearning(trainingData, 15, i, 2)
        validationAccuracyList3.append(majVote(forest, validationData))
        trainingAccuracyList3.append(majVote(forest, trainingData))
        
    validationAccuracyList4 = []
    trainingAccuracyList4 = []
    for x in m:
        forest = RandomForestLearning(trainingData, 15, x, 2)
        validationAccuracyList4.append(majVote(forest, validationData))
        trainingAccuracyList4.append(majVote(forest, trainingData))

    drawMultiple(m, validationAccuracyList3, validationAccuracyList4, ' ',
              'number of trees', 'accuracy', ' ', ' ', 5)
    drawMultiple(m, trainingAccuracyList3, trainingAccuracyList4, ' ',
              'number of trees', 'accuracy', ' ', ' ', 6)