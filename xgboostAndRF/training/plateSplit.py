import cv2
from matplotlib import pyplot as plt

#Function to determine the start and the end of each caracter in the plate
def find_end(start, width, black, black_max, segmentation_spacing):
    end = start + 1
    for m in range(start + 1, width - 1):
      if(black[m] > segmentation_spacing * black_max):
        end = m
        break
    return end

#Function to split the plate caracters and save them in a specific folder
def plateSplit(plateImagePath, imageSegmentsDownloadPath):

    #Ordinary license plate value is 0.95, new energy license plate is changed to 0.9
    segmentation_spacing = 0.9

    white = []  # Record the sum of white pixels in each column
    black = []  # Record the sum of black pixels in each column
    #temporary variables for processing
    white_max = 0
    black_max = 0
    n = 1
    start = 1
    end = 2
    index = 0

    #Image preprocessing
    plate = cv2.imread(plateImagePath)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    ret, thresholdedImage = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    plt.imshow(cv2.cvtColor(thresholdedImage, cv2.COLOR_BGR2RGB)) #Optional : if u want to display the image

    #Image dimensions
    height = thresholdedImage.shape[0] #heigth of the image
    width = thresholdedImage.shape[1] #width of the image

    #Count black and white pixels
    for i in range(width):
      white_count = 0
      black_count = 0
      for j in range(height):
        if thresholdedImage[j][i] == 255:
          white_count += 1
        else:
          black_count += 1

      white.append(white_count)
      black.append(black_count)

    white_max = max(white)
    black_max = max(black)
    
    #Split the image, given the starting point of the character to be split
    while n < width - 1:
      n += 1
      if(white[n] > (1 - segmentation_spacing) * white_max):
        start = n
        end = find_end(start, width, black, black_max, segmentation_spacing)
        n = end
        if end - start > 5:
          character = thresholdedImage[1:height - 20, start:end]
          index+=1
          if(index != 1):
            cv2.imwrite(imageSegmentsDownloadPath + '/' + '{}'.format(index - 1) + '.jpg', character)