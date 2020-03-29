import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getprediction(data, splits, preds):
    first_split = splits[0]
    left_split = splits[1]
    right_split = splits[2]
    pred00 = preds[0]
    pred01 = preds[1]
    pred10 = preds[2]
    pred11 = preds[3]
    prediction = []
    for i in range(data.shape[0]):
        if data[first_split][i] == 0:
            if data[left_split][i] == 0:
                prediction.append(pred00)
            else:
                prediction.append(pred01)
        else:
            if data[right_split][i] == 0:
                prediction.append(pred10)
            else:
                prediction.append(pred11)
    return prediction

def AdaBoostDecisionTree(data, L):
    splits = []
    preds = []
    alphas = []
    D = np.repeat(1 / data.shape[0], data.shape[0])
    for i in range(L):
        first_split, pred0, pred1 = Tree(data, D)
        left = data.loc[data[first_split] == 0]
        right = data.loc[data[first_split] == 1]
        left_split, pred00, pred01 = Tree(left, D)
        right_split, pred10, pred11 = Tree(right, D)
        split = [first_split, left_split, right_split]
        value = [pred00, pred01, pred10, pred11]
        splits.append(split)
        preds.append(value)
        pred = getprediction(data, split, value)
        e = geterror(data, pred, D)
        alpha = 1 / 2 * np.log((1 - e) / e)
        alphas.append(alpha)
        for j in range(len(D)):
            if data['class'][j] != pred[j]:
                D[j] = D[j] * np.exp(alpha)
            else:
                D[j] = D[j] * np.exp(-alpha)
        norm_factor = sum(D)
        D = D / norm_factor
    return splits, preds, alphas

def geterror(data, pred, D):
    e = 0
    for i in range(len(pred)):
        if pred[i] != data['class'][i]:
            e += D[i]
    return e

def AdaBoost(data, L):
    features = []
    pred0s = []
    pred1s = []
    alphas = []
    D = np.repeat(1 / data.shape[0], data.shape[0])
    for i in range(L):
        feature, pred0, pred1 = Tree(data, D)
        features.append(feature)
        pred0s.append(pred0)
        pred1s.append(pred1)

        prediction = []
        for i in range(data.shape[0]):
            if data[feature][i] == 0:
                prediction.append(pred0)
            else:
                prediction.append(pred1)
        e = geterror(data, prediction, D)
        alpha = 1 / 2 * np.log((1 - e) / e)
        alphas.append(alpha)
        for j in range(len(D)):
            if data['class'][j] != prediction[j]:
                D[j] = D[j] * np.exp(alpha)
            else:
                D[j] = D[j] * np.exp(-alpha)
        norm_factor = sum(D)
        D = D / norm_factor

    return features, pred0s, pred1s, alphas

def getvote(predictions, alphas):
    predictions = np.array(predictions)
    vote = []
    for i in range(len(alphas)):
        vote.append(predictions[i] * alphas[i])
    vote = sum(vote)
    for i in range(len(vote)):
        if vote[i] < 0:
            vote[i] = 0
        else:
            vote[i] = 1
    return vote


def AdaBoostPrediction(data, features, pred0s, pred1s, alphas):
    pred = []
    for i in range(len(features)):
        pred0 = pred0s[i]
        pred1 = pred1s[i]
        f = features[i]
        prediction = []
        for j in range(data.shape[0]):
            if data[f][j] == 0:
                prediction.append(pred0)
            else:
                prediction.append(pred1)
        pred.append(np.array(prediction))
        np.place(pred[i], pred[i] == 0, -1)
    return getvote(pred, alphas)

def getaccuracy(data, pred):
    count = 0
    for i in range(len(pred)):
        if data['class'][i] == pred[i]:
            count += 1
    return count / len(pred)

def Tree(data, D):
    U = []
    for i in range(len(data.columns) - 1):
        feature = data.columns[i]
        zeros = data.loc[data[feature] == 0]
        ones = data.loc[data[feature] == 1]
        p0 = sum(D[zeros.index]) / sum(D[data.index])
        p1 = sum(D[ones.index]) / sum(D[data.index])
        U0 = gini(zeros, D)
        U1 = gini(ones, D)
        U.append(p0 * U0 + p1 * U1)
    feature = data.columns[U.index(min(U))]
    zeros = data.loc[data[feature] == 0]
    ones = data.loc[data[feature] == 1]
    zeros0 = zeros.loc[zeros['class'] == 0]
    zeros1 = zeros.loc[zeros['class'] == 1]
    D0 = D[zeros0.index]
    D1 = D[zeros1.index]
    if sum(D0) <= sum(D1):
        pred0 = 1
    else:
        pred0 = 0
    ones0 = ones.loc[ones['class'] == 0]
    ones1 = ones.loc[ones['class'] == 1]
    D0 = D[ones0.index]
    D1 = D[ones1.index]
    if sum(D0) <= sum(D1):
        pred1 = 1
    else:
        pred1 = 0
    return feature, pred0, pred1

def gini(data, D):
    class0 = data.loc[data['class'] == 0]
    class1 = data.loc[data['class'] == 1]
    if not 0 == sum(D[data.index]):
        g = 1 - (sum(D[class0.index]) / sum(D[data.index])) ** 2 - (sum(D[class1.index]) / sum(D[data.index])) ** 2
    else:
        g = 0
    return g

if __name__ == '__main__':
    trainingData = pd.read_csv("pa3_train.csv")
    validationData = pd.read_csv("pa3_val.csv")
    testData = pd.read_csv("pa3_test.csv")
    del trainingData['veil-type_p']
    del validationData['veil-type_p']
    del testData['veil-type_p']

    L = [1, 2, 5, 10, 15]
    train_acc = []
    val_acc = []
    for i in range(len(L)):
        features, pred0s, pred1s, alphas = AdaBoost(trainingData, L[i])
        train_pred = AdaBoostPrediction(trainingData, features, pred0s, pred1s, alphas)
        val_pred = AdaBoostPrediction(validationData, features, pred0s, pred1s, alphas)
        train_acc.append(getaccuracy(trainingData, train_pred))
        val_acc.append(getaccuracy(validationData, val_pred))

    plt.figure("accuracy")
    plt.plot(L, train_acc, label='training')
    plt.plot(L, val_acc, label='validation')
    plt.xlabel('L')
    plt.ylabel('accuracy')
    plt.legend(loc=2)
    plt.show()
    splits, preds, alphas = AdaBoostDecisionTree(trainingData, 6)

    pred = []
    for i in range(len(alphas)):
        pred.append(np.array(getprediction(trainingData, splits[i], preds[i])))
        np.place(pred[i], pred[i] == 0, -1)
    trainingPredictions = getvote(pred, alphas)

    pred = []
    for i in range(len(alphas)):
        pred.append(np.array(getprediction(validationData, splits[i], preds[i])))
        np.place(pred[i], pred[i] == 0, -1)
    validationPredictions = getvote(pred, alphas)

    pred = []
    for i in range(len(alphas)):
        pred.append(np.array(getprediction(testData, splits[i], preds[i])))
        np.place(pred[i], pred[i] == 0, -1)
    testPredictions = getvote(pred, alphas)

    trainingAccuracy = getaccuracy(trainingData, trainingPredictions)
    validationAccuracy = getaccuracy(validationData, validationPredictions)
    np.savetxt("pa3_test.csv", testPredictions)