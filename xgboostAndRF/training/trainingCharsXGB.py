from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.metrics import accuracy_score
import numpy as np
from xgboost import XGBClassifier
import pickle

chars = dl.loadChars()

xchars = [item[1:] for item in chars]
ychars = [item[0] for item in chars]

chars_x_train = []
chars_y_train = []
chars_x_test = []
chars_y_test = []

chars_x_train, chars_x_test, chars_y_train, chars_y_test = train_test_split(xchars, ychars, train_size=0.9)

chars_x_train = np.array(chars_x_train)
chars_x_test = np.array(chars_x_test)
chars_y_train = np.array(chars_y_train)
chars_y_test = np.array(chars_y_test)

xgb = XGBClassifier(n_estimators = 10000)
xgb.fit(chars_x_train, chars_y_train)
y_pred = xgb.predict(chars_x_test)
print(accuracy_score(chars_y_test, y_pred))

dl.displayImage(chars_x_test[20], chars_y_test[20], (32, 32), y_pred[20])
dl.displayImage(chars_x_test[40], chars_y_test[40], (32, 32), y_pred[40])

dl.displayImage(chars_x_test[60], chars_y_test[60], (32, 32), y_pred[60])
dl.displayImage(chars_x_test[80], chars_y_test[80], (32, 32), y_pred[80])

dl.displayImage(chars_x_test[90], chars_y_test[90], (32, 32), y_pred[90])
dl.displayImage(chars_x_test[95], chars_y_test[95], (32, 32), y_pred[95])

dl.displayImage(chars_x_test[20], chars_y_test[20], (32, 32), y_pred[20])
dl.displayImage(chars_x_test[40], chars_y_test[40], (32, 32), y_pred[40])

dl.displayImage(chars_x_test[60], chars_y_test[60], (32, 32), y_pred[60])
dl.displayImage(chars_x_test[80], chars_y_test[80], (32, 32), y_pred[80])

dl.displayImage(chars_x_test[90], chars_y_test[90], (32, 32), y_pred[90])
dl.displayImage(chars_x_test[95], chars_y_test[95], (32, 32), y_pred[95])


pickle.dump(xgb, open("/content/drive/MyDrive/AI/models/xgbCharactersModel.sav", 'wb'))