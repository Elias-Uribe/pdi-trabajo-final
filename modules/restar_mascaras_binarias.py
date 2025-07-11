import cv2
import numpy as np
import matplotlib.pyplot as plt

def restar_mascaras_binarias(mascara_a, mascara_b, mostrar=True):
    """
    Resta dos máscaras binarias: resultado = A - B.

    :param mascara_a: Primera máscara (minuendo), ndarray binaria.
    :param mascara_b: Segunda máscara (sustraendo), ndarray binaria.
    :param mostrar: Si True, muestra la comparación visual.
    :return: Máscara resultante (A sin B).
    """
    try:
        if mascara_a.shape != mascara_b.shape:
            raise ValueError("Las máscaras deben tener el mismo tamaño.")

        if len(mascara_a.shape) != 2 or len(mascara_b.shape) != 2:
            raise ValueError("Las máscaras deben estar en escala de grises (2D).")

        # Asegurar que las máscaras estén en formato uint8 (0 o 255)
        if mascara_a.max() <= 1:
            mascara_a = (mascara_a * 255).astype(np.uint8)
        if mascara_b.max() <= 1:
            mascara_b = (mascara_b * 255).astype(np.uint8)

        # Resta morfológica
        resultado = cv2.subtract(mascara_a, mascara_b)

        # Visualización
        if mostrar:
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(mascara_a, cmap='gray')
            plt.title("Máscara A (minuendo)")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(mascara_b, cmap='gray')
            plt.title("Máscara B (sustraendo)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(resultado, cmap='gray')
            plt.title("Resultado A - B")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return resultado

    except Exception as e:
        print(f"❌ Error al restar máscaras:\n{e}")
        return None
