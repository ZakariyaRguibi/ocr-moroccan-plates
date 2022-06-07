import oneTest
import os

def multipleTest(testDirectory, modelPath, resultDirectory):
  for file in os.listdir(testDirectory):
    testPath = testDirectory + "/" + file
    resultPath = resultDirectory+ "/" + file[-5] 
    oneTest.oneTest(testPath, modelPath, resultPath , file[-5])

#multipleTest("/content/test","/content/drive/MyDrive/AI/RF_XGB_Models/random_forest_model.pbz2", "/content/results")