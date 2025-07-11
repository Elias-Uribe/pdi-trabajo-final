import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt

def cerrar_huecos_con_grietas(mascara_binaria, kernel_size=3, mostrar=True):
    """
    Cierra huecos internos incluso si tienen pequeñas aberturas (grietas).

    1. Dilata ligeramente para cerrar las grietas.
    2. Aplica binary_fill_holes para cerrar huecos internos.
    3. Erosiona para devolver el tamaño original.
    """
    if mascara_binaria.max() == 255:
        mascara_binaria = (mascara_binaria > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Paso 1: dilatar suavemente para cerrar grietas
    dilatada = cv2.dilate(mascara_binaria, kernel, iterations=1)

    # Paso 2: rellenar huecos internos
    rellenada = binary_fill_holes(dilatada).astype(np.uint8)

    # Paso 3: erosionar para revertir la dilatación
    final = cv2.erode(rellenada, kernel, iterations=1) * 255

    if mostrar:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(mascara_binaria * 255, cmap='gray')
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(rellenada * 255, cmap='gray')
        plt.title("Huecos rellenados (con grietas)")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(final, cmap='gray')
        plt.title("Final (sin agrandar células)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return final
