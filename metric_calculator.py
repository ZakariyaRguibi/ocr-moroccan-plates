from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os

import sys
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,"xgboostAndRF")
sys.path.append(mymodule_dir)

mymodule_dir = os.path.join(script_dir,"svm")
sys.path.append(mymodule_dir)
from svm.pretretment import pretreatment, imageToArray, imagePretreatment
import xgboostAndRF.testing.utils as util

def formatDigit(digit):
    if digit>9:
            if digit==10:
               digit="A"
            elif digit==11:
                digit="B"
            elif digit==12:
                digit="D"
            elif digit==13:
                digit="W"
            elif digit==14:
                digit="H"
    return str(digit)

def readFolder(dirPath, foldername=""):
    charArray = []
    for file in os.listdir(dirPath):
        currentImage = os.path.join(dirPath, file)
        charArray.append([foldername,currentImage])
    return charArray

def readChars(directory):
    charArray = []
    for foldername in os.listdir(directory):
        dirPath = os.path.join(directory, foldername)
        if os.path.isdir(dirPath):
            imageArray = readFolder(dirPath, foldername)
        charArray.extend(imageArray)
    return charArray

def loadsvm():
    return util.decompress_pickle("svm/model_dumps/svm_model.pbz2")

def predictLetter(letterPath,clf):
    myletterImage=imagePretreatment(letterPath)
    letter = imageToArray(myletterImage)
    y_pred = clf.predict([letter])
    return formatDigit(y_pred[0])


def folderPredictor(path):
   y_test=[]
   y_pred=[]
   mychars=readChars(path)
   clf=loadsvm()
   for char in mychars:
       y_test.append(char[0])
       y_pred.append(predictLetter(char[1],clf))
   return y_test,y_pred

y_test,y_pred=folderPredictor("dataset/testing2")

confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix\n')
print(confusion)


 #importing accuracy_score, precision_score, recall_score, f1_score

print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

# print('Micro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='micro')))
# print('Micro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='micro')))
# print('Micro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='micro')))

# print('Macro Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='macro')))
# print('Macro Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='macro')))
# print('Macro F1-score: {:.2f}\n'.format(f1_score(y_test, y_pred, average='macro')))

# print('Weighted Precision: {:.2f}'.format(precision_score(y_test, y_pred, average='weighted')))
# print('Weighted Recall: {:.2f}'.format(recall_score(y_test, y_pred, average='weighted')))
# print('Weighted F1-score: {:.2f}'.format(f1_score(y_test, y_pred, average='weighted')))

from sklearn.metrics import classification_report
print('\nClassification Report\n')
print(classification_report(y_test, y_pred, target_names=['0','1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'D', 'W', 'H']))