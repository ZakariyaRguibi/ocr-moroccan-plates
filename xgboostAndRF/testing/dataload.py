#add this script before you start execution in colab
"""from google.colab import drive
drive.mount("/content/drive")"""
#arborescence google drive in colab
"""
---drive
        |---MyDrive
            |---AI
                |---models
                |---train
                    |---chars.csv
                    |---nums.csv
"""
from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv


def loadCsvData(filePath, headerSkips=1, type=int):
    return genfromtxt(filePath, delimiter=',', skip_header=headerSkips, dtype=type)


def displayImage(img, label, dimentions=(28, 28), predicted=""):
    label = str(label)
    predicted = str(predicted)
    img = img.reshape(dimentions)
    plt.imshow(img, cmap="Greys",)

    extraText = "Predicted : " + predicted if predicted else ""
    plt.title("label " + label+" " + extraText)
    plt.show()


def loadChars(testCharPath):
    return loadCsvData(testCharPath, 0)