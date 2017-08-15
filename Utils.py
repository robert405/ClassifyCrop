import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import random as rd

def loadImage(infilename) :
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data

def loadData(allfile):

    files = []
    for file in allfile:
        files += [loadImage(file)]

    return files

def removeMean(imgs):

    #mean = np.mean(img)
    mean = np.mean(imgs, axis=3, keepdims=True)
    mean = np.mean(mean, axis=2, keepdims=True)
    mean = np.mean(mean, axis=1, keepdims=True)

    return (imgs - mean)

def gray(img):

    a = np.mean(img, axis=2, dtype=int)
    a = np.dstack((a, a, a))

    return a

def getCropOfMatrix(img, left, top, shape):

    right = left + shape[0]
    bottom = top + shape[1]
    return img[left:right, top:bottom]

def shiftV(img, nb):

    m,n,o = img.shape

    copyCrop = getCropOfMatrix(img, 0, 0, (nb, n))
    flipCrop = np.flipud(copyCrop)
    imgCrop = getCropOfMatrix(img, 0, 0, ((m - nb), n))
    shiftImg = np.concatenate((flipCrop,imgCrop), axis=0)

    return shiftImg

def brighten(img):

    temp = img * 1.30
    temp[temp > 255] = 255

    return temp


def darken(img):

    return img * 0.70

def blur(imgs):

    blurImgs = []

    for img in imgs:
        bluStr = rd.randint(5, 10)
        blurImgs += [ndimage.gaussian_filter(img, sigma=(bluStr, bluStr, 0), order=0)]

    return blurImgs

def dataAugmentation(data):

    newData = []

    for img in data:
        # flip left right and up down
        leftRight = np.fliplr(img)
        newData += [leftRight]
        newData += [np.flipud(img)]
        newData += [np.flipud(leftRight)]

        # put in gray scale
        newData += [gray(img)]

        # shift verticaly and pad with symetry
        #newData += [shiftV(img, 24)]

        # rotate of 90 degree and 270 degree
        newData += [np.rot90(img)]
        newData += [np.rot90(img, 3)]
        newData += [np.rot90(leftRight)]
        newData += [np.rot90(leftRight, 3)]

        # add brightness and darkness
        newData += [brighten(img)]
        newData += [darken(img)]

    return newData


def getData(goodFiles, badFiles, dataAug, doBlur):

    data = loadData(goodFiles)
    if (dataAug):
        data = data + dataAugmentation(data)

    badData = loadData(badFiles)
    if (dataAug):
        badData = badData + dataAugmentation(badData)

    if (doBlur):
        idx = np.random.randint(len(data), size=2)
        dataToBlur = np.array(data)
        dataToBlur = dataToBlur[idx]
        badData = badData + blur(dataToBlur)

    goodLabel = np.zeros(len(data), dtype=int)
    badLabel = np.ones(len(badData), dtype=int)

    label = np.concatenate((goodLabel, badLabel), axis=0)
    data = data + badData
    data = removeMean(np.array(data))

    shuf = np.arange(len(data))
    np.random.shuffle(shuf)

    data = data[shuf]
    label = label[shuf]

    return data, label