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


def loadChars():
    return loadCsvData("/content/drive/MyDrive/AI/train/chars.csv", 0)