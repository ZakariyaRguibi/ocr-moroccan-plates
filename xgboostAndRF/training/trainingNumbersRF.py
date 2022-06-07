from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

num = dl.loadNums()


xnums = [item[1:] for item in num]
ynums = [item[0] for item in num]

num_x_train = []
num_y_train = []
num_x_test = []
num_y_test = []
num_x_train, num_x_test, num_y_train, num_y_test = train_test_split(xnums, ynums, train_size=0.9)

num_x_train = np.array(num_x_train)
num_y_train = np.array(num_y_train)
num_x_test = np.array(num_x_test)
num_y_test = np.array(num_y_test)


random_forest = RandomForestClassifier(n_estimators = 10000)
random_forest.fit(num_x_train, num_y_train)
y_pred = random_forest.predict(num_x_test)
print(accuracy_score(num_y_test, y_pred))

dl.displayImage(num_x_test[20], num_y_test[20], (28, 28), y_pred[20])
dl.displayImage(num_x_test[40], num_y_test[40], (28, 28), y_pred[40])

dl.displayImage(num_x_test[50], num_y_test[50], (28, 28), y_pred[50])
dl.displayImage(num_x_test[70], num_y_test[70], (28, 28), y_pred[70])


pickle.dump(random_forest, open("/content/drive/MyDrive/AI/models/randomForestNumbersModel.sav", 'wb'))