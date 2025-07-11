import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label
import cv2

def separar_globulos_en_borde(mascara_binaria, mostrar=True):
    """
    Separa una máscara binaria en dos: glóbulos que tocan el borde y los que no.
    Además muestra una visualización comparativa.

    :param mascara_binaria: Imagen binaria (uint8) con glóbulos (valores 0 y 255)
    :param mostrar: Si True, muestra la figura comparativa
    :return: (mascara_interior, mascara_borde)
    """
    # Asegurar que es binaria
    mascara_binaria = (mascara_binaria > 0).astype(np.uint8) * 255

    # Etiquetar todos los objetos
    etiquetas = label(mascara_binaria, connectivity=2)

    # Eliminar objetos que tocan el borde para quedarnos solo con los internos
    etiquetas_sin_borde = clear_border(etiquetas)
    mascara_interior = (etiquetas_sin_borde > 0).astype(np.uint8) * 255

    # Diferencia: los que estaban en el borde
    mascara_borde = cv2.subtract(mascara_binaria, mascara_interior)

    # Mostrar resultados visualmente
    if mostrar:
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Máscara original")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mascara_interior, cmap='gray')
        plt.title("Glóbulos internos")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mascara_borde, cmap='gray')
        plt.title("Glóbulos en el borde")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    return mascara_interior, mascara_borde
