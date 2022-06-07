import oneTest
import os


def multipleTest(testDirectory, modelPath, resultDirectory):
    for file in os.listdir(testDirectory):
        testPath = testDirectory + "/" + file
        resultPath = resultDirectory + "/" + file[-5]
        oneTest.oneTest(testPath, modelPath, resultPath, file[-5])


multipleTest("/home/pandakin/Desktop/test2/test/",
             "/home/pandakin/dev/ocr-moroccan-plates/xgboostAndRF/models/random_forest_model.pbz2", "/home/pandakin/dev/ocr-moroccan-plates/xgboostAndRF/testing/results")
