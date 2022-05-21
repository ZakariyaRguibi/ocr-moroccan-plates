from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
chars = dl.loadChars()
num = dl.loadNums()


xnums = [item[1:] for item in num]
ynums = [item[0] for item in num]

xchars = [item[1:] for item in chars]
ychars = [item[0] for item in chars]

chars_x_train = []
chars_y_train = []
chars_x_test = []
chars_y_test = []

num_x_train = []
num_y_train = []
num_x_test = []
num_y_test = []
num_x_train, num_x_test, num_y_train, num_y_test = train_test_split(
    xnums, ynums, train_size=0.9)


chars_x_train, chars_x_test, chars_y_train, chars_y_test = train_test_split(
    xchars, ychars, train_size=0.9)


num_x_train = np.array(num_x_train)
num_y_train = np.array(num_y_train)
num_x_test = np.array(num_x_test)
num_y_test = np.array(num_y_test)

chars_x_train = np.array(chars_x_train)
chars_x_test = np.array(chars_x_test)
chars_y_train = np.array(chars_y_train)
chars_y_test = np.array(chars_y_test)


""" clf = SVC(kernel='linear')
clf.fit(num_x_train, num_y_train)
y_pred = clf.predict(num_x_test)
print(accuracy_score(num_y_test, y_pred))


dl.displayImage(num_x_test[500], num_y_test[500], (28, 28), y_pred[500])
dl.displayImage(num_x_test[600], num_y_test[600], (28, 28), y_pred[600])

dl.displayImage(num_x_test[700], num_y_test[700], (28, 28), y_pred[700])
dl.displayImage(num_x_test[800], num_y_test[800], (28, 28), y_pred[800])

dl.displayImage(num_x_test[900], num_y_test[900], (28, 28), y_pred[900])
dl.displayImage(num_x_test[400], num_y_test[400], (28, 28), y_pred[400])
 """
clf = SVC(kernel='linear')
clf.fit(chars_x_train, chars_y_train)
y_pred = clf.predict(chars_x_test)
print(accuracy_score(chars_y_test, y_pred))


dl.displayImage(chars_x_test[20], chars_y_test[20], (32, 32), y_pred[20])
dl.displayImage(chars_x_test[40], chars_y_test[40], (32, 32), y_pred[40])

dl.displayImage(chars_x_test[60], chars_y_test[60], (32, 32), y_pred[60])
dl.displayImage(chars_x_test[80], chars_y_test[80], (32, 32), y_pred[80])

dl.displayImage(chars_x_test[90], chars_y_test[90], (32, 32), y_pred[90])
dl.displayImage(chars_x_test[95], chars_y_test[95], (32, 32), y_pred[95])
