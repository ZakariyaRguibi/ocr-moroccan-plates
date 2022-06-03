import csv
import os
import numpy as np
from PIL import Image
from pretretment import *


def imagePretreatment(image):
    # this is for the original database
    # resizedImage = imageResize(image)
    # return resizedImage
    # this is pretretment for test db
    cleanedImage = imageClean(image)
    resizedImage = imageResize(cleanedImage)
    imageBinarized = imageBinarize(resizedImage)
    invertedImage = invertBW(imageBinarized)
    #thinnedImage = imageThin(invertedImage)
    return invertedImage


def getImageArray(imagePath):
    img_file = getImage(imagePath)
    img_file = imagePretreatment(img_file)
    imageArray = imageToArray(img_file)
    return imageArray


def readChars(directory):
    charArray = []
    for foldername in os.listdir(directory):
        dirPath = os.path.join(directory, foldername)
        if os.path.isdir(dirPath):
            for file in os.listdir(dirPath):
                currentImage = os.path.join(dirPath, file)
                imageArray = [foldername]
                imageArray.extend(getImageArray(currentImage))
                charArray.append(imageArray)
    return charArray


def exportCharsToCsv():
    charArray = readChars("./dataset/dataset_characters")
    with open("./dataset/chars.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(charArray)


def reorientText(img):
    # to-do: fix the oriontation of the image
    return img
# exportCharsToCsv()   dim is 32*32
