import csv
import os
import numpy as np
from PIL import Image


def getImageArray(imagePath):
    imageArray = []
    img_file = Image.open(imagePath)
    width, height = img_file.size
    format = img_file.format
    mode = img_file.mode
    # Make image Greyscale
    img_grey = img_file.convert('L')
    # img_grey.save('result.png')
    # img_grey.show()

    # Save Greyscale values
    value = np.asarray(img_grey.getdata(), dtype=int).reshape(
        (img_grey.size[1], img_grey.size[0]))
    value = value.flatten()

    imageArray.extend(value)
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
    charArray = readChars("./dataset/dataset_characters/chars")

   # np.savetxt("img_pixels.csv", charArray, delimiter=',')
    with open("./dataset/chars.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(charArray)


def exportNumToCsv():
    charArray = readChars("./dataset/dataset_characters/nums")

   # np.savetxt("img_pixels.csv", charArray, delimiter=',')
    with open("./dataset/nums.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(charArray)


exportCharsToCsv()  # dim is 32*32
exportNumToCsv()  # dim is 28*28
