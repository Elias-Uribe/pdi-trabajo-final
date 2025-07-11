import cv2
import numpy as np
import matplotlib.pyplot as plt

def dilatar_mascara(mascara_binaria, kernel_size=9, iteraciones=1, mostrar=True):
    """
    Aplica dilatación a una máscara binaria para expandir regiones blancas.

    :param mascara_binaria: Imagen binaria (np.ndarray)
    :param kernel_size: Tamaño del elemento estructurante (debe ser impar).
    :param iteraciones: Cuántas veces aplicar la operación.
    :return: Máscara dilatada (np.ndarray)
    """
    try:
        if mascara_binaria.ndim != 2:
            raise ValueError("La máscara debe ser una imagen binaria en escala de grises.")

        # Elemento estructurante elíptico
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # Aplicar dilatación
        mascara_dilatada = cv2.dilate(mascara_binaria, kernel, iterations=iteraciones)

        if mostrar:
            # Mostrar visualización
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(mascara_binaria, cmap='gray')
            plt.title("Máscara Original")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mascara_dilatada, cmap='gray')
            plt.title(f"Máscara Dilatada (k={kernel_size})")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return mascara_dilatada

    except Exception as e:
        print(f"❌ Error al dilatar máscara: {e}")
        return None