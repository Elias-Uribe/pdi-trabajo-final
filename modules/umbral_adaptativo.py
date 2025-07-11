import cv2
import numpy as np
import matplotlib.pyplot as plt

def umbral_adaptativo(imagen_gris, bloque=11, C=2, mostrar=True):
    """
    Aplica umbralización adaptativa a una imagen en escala de grises.
    
    :param imagen_gris: Imagen 2D en escala de grises (np.ndarray)
    :param bloque: Tamaño del vecindario (debe ser impar y >1)
    :param C: Constante que se resta del valor medio local
    :param mostrar: Si True, muestra el resultado
    :return: Máscara binaria (0 y 255)
    """
    if imagen_gris.dtype != np.uint8:
        imagen_gris = cv2.normalize(imagen_gris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    mascara = cv2.adaptiveThreshold(
        imagen_gris,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,  # o cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=bloque,
        C=C
    )

    if mostrar:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.title("Imagen Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mascara, cmap='gray')
        plt.title("Umbralización Adaptativa")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mascara
