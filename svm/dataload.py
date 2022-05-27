from numpy import genfromtxt
import matplotlib.pyplot as plt
import csv


def loadCsvData(filePath, headerSkips=1, type=int):
    """load a csv file 

     Parameters:
        filePath (str): The path of the csv file
        headerSkips (int):The number of lines to skip at the beginning of the file.
        type (class): DataType of the csv values {int,float, String}
     Returns:
        array
    """

    return genfromtxt(filePath,
                      delimiter=',', skip_header=headerSkips, dtype=type)


def displayImage(img, label, dimentions=(32, 32), predicted=""):
    """Display a DataImage """
    label = str(label)
    predicted = str(predicted)
    img = img.reshape(dimentions)
    plt.imshow(img, cmap="Greys",)

    extraText = "Predicted : " + predicted if predicted else ""
    plt.title("label " + label+" " + extraText)
    plt.show()


def loadChars():
    return loadCsvData("dataset/chars.csv", 0)
