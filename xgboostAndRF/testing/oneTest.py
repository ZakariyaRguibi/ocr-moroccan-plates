import testing
import pretreatmentAndConversion as pretreatment

def oneTest(imagePath, modelPath, savingResultsPath, label):
  pretreatment.imagePretreatmentAndConversionToCSV(imagePath, savingResultsPath, label)
  testing.test(modelPath, savingResultsPath)

#oneTest("/content/testing/num0.png", "/content/drive/MyDrive/AI/RF_XGB_Models/random_forest_model.pbz2", "/content/results/res0.csv", 0)
#pretreatment.imagePretreatmentAndConversionToCSV("/content/num2.png", "/content/result2.csv", 2)
#test.test("/content/drive/MyDrive/AI/RF_XGB_Models/random_forest_model.pbz2", "/content/result2.csv")