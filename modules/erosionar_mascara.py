import cv2
import numpy as np
import matplotlib.pyplot as plt

def erosionar_mascara(mascara_binaria, kernel_size=9, iteraciones=1, mostrar=True):
    """
    Aplica erosión a una máscara binaria para reducir regiones blancas.

    :param mascara_binaria: Imagen binaria (np.ndarray)
    :param kernel_size: Tamaño del elemento estructurante (debe ser impar).
    :param iteraciones: Cuántas veces aplicar la operación.
    :return: Máscara erosionada (np.ndarray)
    """
    try:
        if mascara_binaria.ndim != 2:
            raise ValueError("La máscara debe ser una imagen binaria en escala de grises.")

        # Elemento estructurante elíptico
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Aplicar erosión
        mascara_erosionada = cv2.erode(mascara_binaria, kernel, iterations=iteraciones)

        if mostrar:
            # Mostrar visualización
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(mascara_binaria, cmap='gray')
            plt.title("Máscara Original")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mascara_erosionada, cmap='gray')
            plt.title(f"Máscara Erosionada (k={kernel_size})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return mascara_erosionada

    except Exception as e:
        print(f"❌ Error al erosionar máscara: {e}")
        return None
