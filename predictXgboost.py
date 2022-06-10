import os
import sys


script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir,"xgboostAndRF")
sys.path.append(mymodule_dir)
mymodule_dir = os.path.join(script_dir,"platesplit")
sys.path.append(mymodule_dir)
mymodule_dir = os.path.join(script_dir,"svm")
sys.path.append(mymodule_dir)

import svm.dataload as dl
import numpy as np
import svm.utils as util
import svm.imageToCsv as pt
from platesplit.platesplit import findLetters
from svm.pretretment import pretreatment, imageToArray




def predict(plateImagePath):
    # load model
    clf = util.decompress_pickle("xgboostAndRF/models/XGBoost_model.pbz2")
    size = (32, 32)
    letters = findLetters(plateImagePath)
    predictionResult = []
    for letter in letters:
        #format image to the uint8 format instead of the float format
        letter = (letter*255).astype(np.uint8)
        letter = imageToArray(pretreatment(letter))
        y_pred = clf.predict([letter])
        imageArray = np.array([letter])
        #dl.displayImage(imageArray[0], "", size, y_pred[0]).
        predictionResult.append( y_pred[0])
    print(str(predictionResult))



predict("/home/pandakin/dev/ocr-moroccan-plates/dataset/test/plates/plat3.png")
