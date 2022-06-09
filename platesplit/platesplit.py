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


def myFoo():
    # testing only
    image_path = "/home/pandakin/Desktop/123.png"
    LpImg = [preprocess_image(image_path)]

    if True:  # (len(LpImg)): #Vérifier si qu'une plaque au moins est bien détectée

        # Convertir le résultat sur un échelle de 8 bits
        plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

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

    # visualiser les résultats des différents étapes
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

    # Détecter l'ensemble des contours dans l'image sous la forme Bounding Box
    cont, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Créer une copie de l'image pour dessiner les contours
    test_plat = plate_image.copy()

    # Initialiser une liste qui va contenir les contours des caractères
    crop_characters = []

    # définir la longeur et largeur standarisées des caractères
    digit_w, digit_h = 30, 60

    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w

        # Ne selectionner que les contours avec le ratio défini
        if 1 <= ratio <= 8:

            # Ne selectionner que les contours dont la longeur occupe 20% de la largeur de l'image de la plaque
            if h/plate_image.shape[0] >= 0.5:

                # Dessiner les rectangles autour des caractères détectés
                cv2.rectangle(test_plat, (x, y),
                              (x + w, y + h), (0, 255, 0), 2)
                curr_num = thre_mor[y:y+h, x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(
                    curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

    print("Detect {} letters...".format(len(crop_characters)))
    fig = plt.figure(figsize=(10, 6))
    plt.axis(False)
    plt.imshow(test_plat)
    plt.show()


myFoo()
