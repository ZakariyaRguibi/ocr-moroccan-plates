from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
num = dl.loadNums()


xnums = [item[2:] for item in num]
ynums = [item[0:2] for item in num]


num_x_train = []
num_y_train = []
num_x_test = []
num_y_test = []
num_x_train, num_x_test, num_y_train, num_y_test = train_test_split(
    xnums, ynums, train_size=0.9)


num_x_train = np.array(num_x_train)
num_y_train = np.array(num_y_train)
num_x_test = np.array(num_x_test)
num_y_test = np.array(num_y_test)

locations = [item[0] for item in num_y_test]
num_y_test = [item[1] for item in num_y_test]

num_y_train = [item[1] for item in num_y_train]


clf = SVC(kernel='linear')
clf.fit(num_x_train, num_y_train)
y_pred = clf.predict(num_x_test)


print(accuracy_score(num_y_test, y_pred))

print(locations[500])
dl.displayImage(num_x_test[500], num_y_test[500], (28, 28), y_pred[500])
print(locations[600])

dl.displayImage(num_x_test[600], num_y_test[600], (28, 28), y_pred[600])

dl.displayImage(num_x_test[700], num_y_test[700], (28, 28), y_pred[700])
print(locations[700])

dl.displayImage(num_x_test[800], num_y_test[800], (28, 28), y_pred[800])
print(locations[800])

dl.displayImage(num_x_test[900], num_y_test[900], (28, 28), y_pred[900])
print(locations[900])

dl.displayImage(num_x_test[400], num_y_test[400], (28, 28), y_pred[400])
print(locations[400])
