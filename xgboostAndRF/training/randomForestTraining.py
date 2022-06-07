from sklearn.model_selection import train_test_split
import dataload as dl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import compress

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


random_forest = RandomForestClassifier(n_estimators = 10000)
random_forest.fit(chars_x_train, chars_y_train)

compress.compressed_pickle("/content/drive/MyDrive/AI/models/random_forest_model", random_forest)

y_pred = random_forest.predict(chars_x_test)
print(accuracy_score(chars_y_test, y_pred))

# accuracy : 0.9773071104387292