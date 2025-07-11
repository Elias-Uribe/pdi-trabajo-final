import cv2
import matplotlib.pyplot as plt

def cerrar_huecos_morfologicos(mascara_binaria, kernel_size=5, mostrar=False):
    """
    Aplica cierre morfológico para rellenar huecos pequeños dentro de objetos.

    :param mascara_binaria: Imagen binaria (np.ndarray con valores 0 y 255)
    :param kernel_size: Tamaño del kernel estructurante (debe ser impar).
    :param mostrar: Si True, muestra antes y después.
    :return: Máscara con huecos cerrados.
    """
    try:
        if mascara_binaria.ndim != 2:
            raise ValueError("La máscara debe ser en escala de grises (binaria).")

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mascara_cerrada = cv2.morphologyEx(mascara_binaria, cv2.MORPH_CLOSE, kernel)

        if mostrar:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(mascara_binaria, cmap='gray')
            plt.title("Antes (huecos visibles)")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(mascara_cerrada, cmap='gray')
            plt.title("Después (huecos cerrados)")
            plt.axis('off')
            plt.tight_layout()
            plt.show()

        return mascara_cerrada

    except Exception as e:
        print(f"❌ Error al cerrar huecos morfológicos: {e}")
        return mascara_binaria  # En caso de error, se devuelve sin cambios