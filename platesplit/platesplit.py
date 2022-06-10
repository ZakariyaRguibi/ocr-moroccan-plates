import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Fonction pour ordonner les contours selon leurs postions de la gauche vers la droite


def sort_contours(cnts, reverse=False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts


def preprocess_image(image_path, resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224, 224))
    return img


def exportImage(image, destination, filename):
    cv2.imwrite(destination + filename, image)


def exportImageArrays(imageArray, folder, id=""):
    index = 0
    for image in imageArray:
        image = (image*255).astype(np.uint8)
        exportImage(image, folder, id+str(index)+".jpg")
        index = index + 1


def cutContours(image, contourArray):
    # returns an array of cutted images
    cutContours = []
    for box in contourArray:
        (x, y, w, h) = box[0], box[1], box[2], box[3]
        cutContours.append(image[y:y+h, x:x+w])
    return cutContours


def imagePreTretment(image):
    # Convertir le résultat sur un échelle de 8 bits
    plate_image = cv2.convertScaleAbs(image, alpha=(255.0))

    # Convertir l'image en un gradient de couleur grise
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

    # Appliquer un lissage gaussien
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Appliquer un treshold de 180
    binary = cv2.threshold(blur, 180, 255,
                           cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Appliquer la dilatation
    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
    #plotImages(plate_image, gray, blur, binary, thre_mor)
    return thre_mor


def plotImages(plate_image, gray, blur, binary, thre_mor):
    # visualiser les résultats des différents étapes de pretretment
    fig = plt.figure(figsize=(12, 7))
    plt.rcParams.update({"font.size": 18})
    grid = gridspec.GridSpec(ncols=2, nrows=3, figure=fig)
    plot_image = [plate_image, gray, blur, binary, thre_mor]
    plot_name = ["1-Image Originale", "2-Image grise",
                 "3-Image après lissage gaussien", "4-Binary image", "5-Image après dilatation"]

    for i in range(len(plot_image)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.title(plot_name[i])
        if i == 0:
            plt.imshow(plot_image[i])
        else:
            plt.imshow(plot_image[i], cmap="gray")
    plt.show()


def showBondingboxes(myImage, bounding):
    image =myImage.copy()
    for box in bounding:
        (x, y, w, h) = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    fig = plt.figure(figsize=(10, 6))
    plt.axis(False)
    plt.imshow(image)
    plt.show()


def findLetters(image_path):
    myImage = preprocess_image(image_path)
    thre_mor = imagePreTretment(myImage)
    # Détecter l'ensemble des contours dans l'image sous la forme Bounding Box
    cont, _ = cv2.findContours(
        thre_mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Initialiser une liste qui va contenir les contours des caractères
    bondingBox = []
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        # Ne selectionner que les contours avec le ratio défini
        if 0.5 <= ratio <= 8:
            # Ne selectionner que les contours dont la longeur occupe 20% de la largeur de l'image de la plaque
            if h/myImage.shape[0] >= 0.2:
                bondingBox.append([x, y, w, h])
    print("Detect {} letters...".format(len(bondingBox)))
    # show image with rectangles
    showBondingboxes(myImage, bondingBox)

    return cutContours(myImage, bondingBox)


#image_path = "/home/pandakin/Desktop/9.jpg"
#letters = findLetters(image_path)
