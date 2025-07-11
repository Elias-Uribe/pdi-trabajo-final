import cv2
import matplotlib.pyplot as plt
import numpy as np

def umbral_otsu(imagen_gris, invertir=False, mostrar=True):
    """
    Aplica umbralización binaria automática usando el método de Otsu.

    :param imagen_gris: Imagen en escala de grises (np.ndarray)
    :param invertir: Si True, invierte la máscara binaria
    :param mostrar: Si True, muestra la imagen y la máscara
    :return: Máscara binaria (0 y 255), valor de umbral encontrado
    """
    if imagen_gris.dtype != np.uint8:
        imagen_gris = cv2.normalize(imagen_gris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    tipo_umbral = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY

    umbral, mascara = cv2.threshold(imagen_gris, 0, 255, tipo_umbral + cv2.THRESH_OTSU)

    if mostrar:
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(imagen_gris, cmap='gray')
        plt.title("Imagen original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mascara, cmap='gray')
        plt.title(f"Máscara Otsu (T={umbral:.2f})")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return mascara, umbral
