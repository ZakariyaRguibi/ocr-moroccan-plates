from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import utils

size = (32, 32)

chars = dl.loadChars()

xchars = [item[1:] for item in chars]
ychars = [item[0] for item in chars]

chars_x_train = []
chars_y_train = []
chars_x_test = []
chars_y_test = []


chars_x_train, chars_x_test, chars_y_train, chars_y_test = train_test_split(
    xchars, ychars, train_size=0.9)


chars_x_train = np.array(chars_x_train)
chars_x_test = np.array(chars_x_test)
chars_y_train = np.array(chars_y_train)
chars_y_test = np.array(chars_y_test)


clf = SVC(kernel='linear')
clf.fit(chars_x_train, chars_y_train)

utils.compressed_pickle("svm/model_dumps/svm_model", clf)

clf = utils.decompress_pickle("svm/model_dumps/svm_model.pbz2")

y_pred = clf.predict(chars_x_test)
print(accuracy_score(chars_y_test, y_pred))

"""
dl.displayImage(chars_x_test[500], chars_y_test[500], size, y_pred[500])
dl.displayImage(chars_x_test[600], chars_y_test[600], size, y_pred[600])

dl.displayImage(chars_x_test[700], chars_y_test[700], size, y_pred[700])
dl.displayImage(chars_x_test[985], chars_y_test[985], size, y_pred[985])

dl.displayImage(chars_x_test[900], chars_y_test[900], size, y_pred[900])
dl.displayImage(chars_x_test[400], chars_y_test[400], size, y_pred[400])



\"""
   clf = SVC(kernel='linear')
   clf.fit(chars_x_train, chars_y_train)
   y_pred = clf.predict(chars_x_test)
   print(accuracy_score(chars_y_test, y_pred))
   \"""
"""
