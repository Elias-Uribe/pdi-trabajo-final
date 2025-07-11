import cv2
import numpy as np
import matplotlib.pyplot as plt

def umbralizar_binario_opencv(imagen, umbral=128, invertir=False):
    """
    Aplica un umbral binario usando OpenCV.

    Parámetros:
        - imagen (str o np.ndarray): Ruta o imagen en escala de grises.
        - umbral (int): Valor de umbral (0 a 255).
        - invertir (bool): Si True, invierte el resultado (negro ↔ blanco).

    Retorna:
        - np.ndarray: Imagen binarizada (0 y 255).
    """
    try:
        # 📌 Cargar imagen si es una ruta
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen, cv2.IMREAD_GRAYSCALE)

        if imagen is None or len(imagen.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises.")

        # 📌 Tipo de umbral
        tipo = cv2.THRESH_BINARY_INV if invertir else cv2.THRESH_BINARY

        # 📌 Aplicar umbral
        _, imagen_binaria = cv2.threshold(imagen, umbral, 255, tipo)

        # 📌 Mostrar resultados
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(imagen, cmap='gray')
        axes[0].set_title("Imagen Original")
        axes[0].axis("off")

        axes[1].imshow(imagen_binaria, cmap='gray')
        axes[1].set_title(f"Umbral binario (>{umbral})")
        axes[1].axis("off")

        plt.show()

        return imagen_binaria

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def umbralizar_por_rango_opencv(imagen, umbral_min=50, umbral_max=150, invertir=False):
    """
    Aplica una umbralización por rango usando OpenCV, con opción de invertir la máscara.

    :param imagen: Ruta o imagen en escala de grises (np.ndarray).
    :param umbral_min: Límite inferior del umbral.
    :param umbral_max: Límite superior del umbral.
    :param invertir: Si True, invierte la máscara resultante (útil para contar objetos negros).
    :return: Imagen binarizada (uint8) con valores 0 y 255.
    """
    try:
        # Leer si es ruta
        if isinstance(imagen, str):
            imagen = cv2.imread(imagen, cv2.IMREAD_GRAYSCALE)

        if imagen is None or len(imagen.shape) != 2:
            raise ValueError("La imagen debe estar en escala de grises.")

        # Aplicar umbral por rango
        mascara = cv2.inRange(imagen, umbral_min, umbral_max)

        # Invertir si es necesario
        if invertir:
            mascara = cv2.bitwise_not(mascara)

        # Mostrar resultado
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(imagen, cmap='gray')
        plt.title("Imagen original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mascara, cmap='gray')
        titulo = f"Rango [{umbral_min}, {umbral_max}]"
        if invertir:
            titulo += " (Invertido)"
        plt.title(titulo)
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return mascara

    except Exception as e:
        print(f"❌ Error al umbralizar por rango:\n{e}")
        return None

