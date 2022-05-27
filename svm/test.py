from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import utils as util
import imageToCsv as pt

size = (32, 32)

clf = util.decompress_pickle("svm/model_dumps/svm_model.pbz2")

testFolder = "dataset/test"


directoryArray = pt.readChars(testFolder)
xnums = [item[1:] for item in directoryArray]
xnums = np.array(xnums)

y_pred = clf.predict(xnums)

for i in range(0, len(xnums)):
    dl.displayImage(xnums[i], "xnums["+str(i)+"]", size, y_pred[i])
