from collections import defaultdict
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import random
import itertools
import operator
from collections import Counter
from builtins import print, len

from io import open
from skimage.feature import hog
from skimage import data, color, exposure
from skimage.transform import rescale, resize
import time


def imread(loc):
    img = mpimg.imread(loc)
    return img


def readTxt(filename):
    txt = []
    with open(filename) as f:
        for line in f:
            line = line.rstrip("\n\r")
            txt.append(line.split(" "))
        f.close()
    return txt


def ColorHistogram(rgbarray, numofbin=256):
    histR, bins = np.histogram(rgbarray[:, 0], np.arange(0, numofbin + 1), density=True)
    histG, bins = np.histogram(rgbarray[:, 1], np.arange(0, numofbin + 1), density=True)
    histB, bins = np.histogram(rgbarray[:, 2], np.arange(0, numofbin + 1), density=True)
    hist = np.concatenate((histR, histG, histB), axis=0)
    return hist


def maskedImgMaker(image, imageLoc):
    annotation = ("./annotations/annotations-mat/" + imageLoc.rstrip(".jpg") + ".mat")
    birdmat = sio.loadmat(annotation)
    segment = birdmat['seg']
    maskedImg = image * np.expand_dims(segment, axis=2)

    return maskedImg


def bboxImgMaker(image, imageLoc):
    annotation = ("./annotations/annotations-mat/" + imageLoc.rstrip(".jpg") + ".mat")
    birdmat = sio.loadmat(annotation)
    bounding_box = birdmat['bbox']
    bottom = bounding_box['bottom'][0][0][0][0]
    top = bounding_box['top'][0][0][0][0]
    left = bounding_box['left'][0][0][0][0]
    right = bounding_box['right'][0][0][0][0]
    bbox_image = image[top:bottom, left:right, :]

    return bbox_image


def filteredImageHistogram(imageLoc):
    image = imread("./images/images/" + imageLoc)
    maskedImg = maskedImgMaker(image, imageLoc)
    bbox_image = bboxImgMaker(maskedImg, imageLoc)
    hist = ColorHistogram(bbox_image)

    return hist


def colorHistDataMaker(testDict, trainDict):
    testColorHistograms, trainColorHistograms = [], []

    for i in testDict.keys():
        testColorHistograms.append(filteredImageHistogram(testDict.get(i)[1]))
    for i in trainDict.keys():
        trainColorHistograms.append(filteredImageHistogram(trainDict.get(i)[1]))
    return np.array(testColorHistograms), np.array(trainColorHistograms)


def dataMaker(testTxt, trainTxt):
    testAttributes, trainAttributes = [], []
    testDict, trainDict = defaultdict(list), defaultdict(list)

    for i in range(len(testTxt)):
        testAttributes.append(attributes[int(testTxt[i][0])])
        testDict[i].append(testTxt[i][1].split("/", 1)[0])
        testDict[i].append(testTxt[i][1])
    for i in range(len(trainTxt)):
        trainAttributes.append(attributes[int(trainTxt[i][0])])
        trainDict[i].append(trainTxt[i][1].split("/", 1)[0])
        trainDict[i].append(trainTxt[i][1])

    return np.array(testAttributes), np.array(trainAttributes), testDict, trainDict


def kNNfunc(testData, trainData, kVal):
    distance = trainData - testData
    distance = np.power(distance, 2)
    distance = np.sum(distance, axis=1)

    neighbours = np.argsort(distance)[:kVal]

    return neighbours


def weightedkNNfunc(testData, trainData, kVal):
    distance = trainData - testData
    distance = np.power(distance, 2)
    distance = np.sum(distance, axis=1)

    neighbours = np.argsort(distance)[:kVal]
    sortedDistance = np.sort(distance)[:kVal]

    return neighbours, sortedDistance


def accuracy(correctResult, testDict):
    correctPre = 0

    for i in range(len(testDict)):
        testVal = testDict[i][0]
        predict = Counter(correctResult[i]).most_common(1)[0][0]
        if testVal == predict:
            correctPre += 1
    return (correctPre / len(testDict)) * 100


def weightedAccuracy(result, testDict):
    correctPre = 0

    for i in range(len(testDict)):
        predict = max(result[i], key=lambda key: result[i][key])
        testVal = testDict[i][0]

        if testVal == predict:
            correctPre += 1
    return (correctPre / len(testDict)) * 100


def splitData(data, i, partition):
    q1 = i * round(partition)
    q2 = (i + 1) * round(partition)

    test1 = data[q1:q2]
    train1 = data[0:q1]
    for k in data[q2:]:
        train1.append(k)

    return test1, train1


def weightClasses(neighbours, weights, trainDict):
    weightOrder = []
    for i in range(len(neighbours)):
        order = defaultdict(float)
        for k in range(len(neighbours[i])):
            itemweight = weights[i][k]
            if itemweight == 0:
                itemweight = 1
            dataclass = trainDict[neighbours[i][k]]
            if order[dataclass[0]] is None:
                order[dataclass[0]] = (1 / np.power(itemweight, 2))
            else:
                order[dataclass[0]] += (1 / np.power(itemweight, 2))
        weightOrder.append(order)

    return weightOrder


def kFoldCross(dataTxt, kFoldNum, kNum, implementation):
    partition = len(dataTxt) / kFoldNum

    successRates = []

    for i in range(kFoldNum):

        testTxt, trainTxt = splitData(dataTxt, i, partition)

        if implementation == "kNN":
            successRates.append(kNNPart(testTxt, trainTxt, kNum))

        elif implementation == "w-kNN":
            successRates.append(weightedkNNPart(testTxt, trainTxt, kNum))

    return successRates


def kNNPart(testTxt, trainTxt, kNum):
    testAttributes, trainAttributes, testDict, trainDict = dataMaker(testTxt, trainTxt)
    # delete for color Histogram
    # testColorHistograms, trainColorHistograms = colorHistDataMaker(testDict,trainDict)
    attributeNeighbours = []
    colorHistNeightbours = []

    attributeClasses = []
    colorHistClasses = []

    for testIndex in testDict.keys():
        # you can delete # for calculate with color Histogram
        attributeNeighbours.append(kNNfunc(testAttributes[testIndex], trainAttributes, kNum))
        # colorHistNeightbours.append(kNNfunc(testColorHistograms[testIndex], trainColorHistograms, kNum))

    for x in attributeNeighbours:
        attClass = []
        for k in x:
            attClass.append(trainDict.get(k)[0])
        attributeClasses.append(attClass)

    return accuracy(attributeClasses, testDict)


def weightedkNNPart(testTxt, trainTxt, kNum):
    testAttributes, trainAttributes, testDict, trainDict = dataMaker(testTxt, trainTxt)
    # delete for color Histogram
    # testColorHistograms, trainColorHistograms = colorHistDataMaker(testDict,trainDict)
    attributeNeighbours = []
    # colorHistNeightbours = []

    distance = []
    for testIndex in testDict.keys():
        n1, n2 = weightedkNNfunc(testAttributes[testIndex], trainAttributes, kNum)
        # n1, n2 = weightedkNNfunc(testColorHistograms[testIndex], trainColorHistograms, kNum)
        attributeNeighbours.append(n1)
        distance.append(n2)

    weightOrder = weightClasses(attributeNeighbours, distance, trainDict)

    return weightedAccuracy(weightOrder, testDict)


def main():
    kNum = 21
    kFoldNum = 10

    # you can change "kNN" to "w-kNN" for calculating with weighted - kNN algorithm
    successRates = kFoldCross(dataTxt, kFoldNum, kNum, "kNN")

    mean = sum(successRates) / float(len(successRates))

    print("Accuracy is: " + str(mean) + ". Data size = " + str(len(dataTxt)) + ". k-fold = " + str(kFoldNum) + " kNN =" + str(kNum))


attributes = np.load('attributes.npy')
dataTxt = readTxt("train.txt")
main()
