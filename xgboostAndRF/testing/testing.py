import dataload as dl
import numpy as np
import utils as compress

def test(modelPath, charPathCSV):

  size = (32, 32)
  chars = dl.loadChars(charPathCSV)

  xchars = np.array(chars[1:])
  ychars = np.array(chars[0])
  random_forest = compress.decompress_pickle(modelPath)

  y_pred = random_forest.predict(xchars.reshape(1, -1))
  dl.displayImage(xchars.reshape(1, -1), ychars, size, y_pred)