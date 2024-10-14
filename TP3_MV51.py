import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

zoom = 0.8  # Facteur de zoom

# Charger l'image
I1 = cv.imread("D:/Documents/INFO5/VA51/TP3_enonce/I2filter.png", cv.IMREAD_GRAYSCALE)
I2 = cv.imread("D:/Documents/INFO5/VA51/TP3_enonce/Ibin.png", cv.IMREAD_GRAYSCALE)

def Q1(): #application filtres
    H1 = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]])
    #retire hautes fréquences (contours car passage brusque)

    H2 = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]])
    #contours verticaux

    H3 = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]])
    #chaque contour est doublé

    # Afficher l'image d'origine et les images filtrées
    cv.imshow('Original Image', I1)
    cv.imshow('Filtered moyenneur', cv.filter2D(I1, -1, H1))
    cv.imshow('Filtered gaussian', cv.filter2D(I1, -1, H2))
    cv.imshow('Filtered Laplac', cv.filter2D(I1, -1, H3))
    cv.waitKey(0)
    cv.destroyAllWindows()

def Q3(): #application bordure
    H1 = np.array([
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9],
        [1 / 9, 1 / 9, 1 / 9]])

    cv.imshow('Original Image', I1)
    cv.imshow('Bordure constant', cv.filter2D(I1, -1, H1, borderType=cv.BORDER_CONSTANT))
    cv.imshow('Bordure replicat', cv.filter2D(I1, -1, H1, borderType=cv.BORDER_REPLICATE))
    cv.waitKey(0)
    cv.destroyAllWindows()

def Q4():
    """si on augmente la taille du filtre, l"image sera plus floue par exemple ici"""
    H1 = matrix = np.ones((5, 5)) * (1/25)

    cv.imshow('Filtered moyenneur', cv.filter2D(I1, -1, H1))
    cv.waitKey(0)
    cv.destroyAllWindows()

def filtre_median(): #retire les bruits de l'image
    cv.imshow('Filtre median', cv.medianBlur(I1, 5))
    cv.waitKey(0)
    cv.destroyAllWindows()

def erosion():  #application filtre erosion
    H = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], np.uint8)
    cv.imshow('image origine', I2)
    cv.imshow('Image erodee', cv.erode(I2, H))
    cv.waitKey(0)
    cv.destroyAllWindows()

def dilate():
    H = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], np.uint8)
    cv.imshow('image origine', I2)
    cv.imshow('Image dilatee', cv.dilate(I2, H))
    cv.waitKey(0)
    cv.destroyAllWindows()

def ouverture(): #erosion puis dilatation
    SE1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    cv.imshow('image origine', I2)
    cv.imshow('Image apres ouverture', cv.erode(cv.dilate(I2, SE1),SE1))
    cv.waitKey(0)
    cv.destroyAllWindows()

def fermeture(): #dilatation puis erosion
    SE1 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (12, 12))
    cv.imshow('image origine', I2)
    cv.imshow('Image apres fermeture', cv.dilate(cv.erode(I2, SE1), SE1))
    cv.waitKey(0)
    cv.destroyAllWindows()

def nagao_filter(image):

    output_image = np.zeros_like(image)

    kernels = [
        np.array([[1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]),
        np.array([[1, 1, 1, 1, 1],
                  [0, 1, 1, 1, 0],
                  [0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]),
        np.array([[0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0]]),
        np.array([[1, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0]]),
        np.array([[0, 0, 0, 0, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 0, 1, 1],
                  [0, 0, 0, 0, 1]]),
        np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0],
                  [1, 1, 1, 0, 0]]),
        np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 0],
                  [1, 1, 1, 1, 1]]),
        np.array([[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1],
                  [0, 0, 1, 1, 1]])
    ]

    for i in range(2, image.shape[0] - 2):
        for j in range(2, image.shape[1] - 2):
            variances = []
            best_kernel = None
            region = image[i - 2:i + 3, j - 2:j + 3]

            for kernel in kernels:
                if np.sum(kernel) != 0:
                    masked_region = region[kernel == 1]
                    variance = np.var(masked_region)
                    variances.append(variance)
                else:
                    variances.append(np.inf)

            best_kernel = kernels[np.argmin(variances)]

            if np.sum(best_kernel) != 0:
                best_masked_region = region[best_kernel == 1]
                mean_value = np.mean(best_masked_region)
                output_image[i, j] = mean_value
            else:
                output_image[i, j] = 0

    cv.imshow("Image Originale", image)
    cv.imshow("Image Filtre Nagao", output_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

# Redimensionner l'image
width = int(I1.shape[1] * zoom)
height = int(I1.shape[0] * zoom)
dim = (width, height)
resized_image = cv.resize(I1, dim, interpolation=cv.INTER_AREA)

#questions

#Q2()
#Q3()
#Q4()
#filtre_median()
#erosion()
#dilate()
#ouverture()
#fermeture()

# Appliquer le filtre de Nagao
# Appliquer le filtre de Nagao
filtered_image = nagao_filter(I1)

