import cv2
import numpy as np
import matplotlib.pyplot as plt

def extraer_objeto_principal(mascara_binaria, imagen_original=None):
    """
    Detecta y extrae el objeto más grande de una imagen binaria.
    Opcionalmente lo aplica sobre la imagen origina.
    
    :param mascara_binaria: Imagen binaria con 0 y 255.
    :param imagen_original: Imagen original (para aplicar máscara). Opcional.
    :return: Máscara del objeto principal y recorte aplicado si se da la original.
    """
    try:
        # Asegurarse que la máscara sea uint8
        if mascara_binaria.dtype != np.uint8:
            mascara_binaria = mascara_binaria.astype(np.uint8)

        # Encontrar contornos
        contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            print("❌ No se encontraron contornos.")
            return None, None

        # Encontrar el contorno más grande (por área)
        contorno_mayor = max(contornos, key=cv2.contourArea)

        # Crear una nueva máscara solo con ese contorno
        mascara_objeto = np.zeros_like(mascara_binaria)
        cv2.drawContours(mascara_objeto, [contorno_mayor], -1, 255, thickness=-1)

        # Si se da una imagen original, aplicamos la máscara
        imagen_aplicada = None
        if imagen_original is not None:
            # Si es RGB, multiplicamos canal por canal
            if imagen_original.ndim == 3:
                imagen_aplicada = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_objeto)
            else:
                imagen_aplicada = cv2.bitwise_and(imagen_original, imagen_original, mask=mascara_objeto)

        # Mostrar resultado
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(mascara_objeto, cmap='gray')
        plt.title("Máscara: objeto más grande")
        plt.axis("off")

        if imagen_aplicada is not None:
            plt.subplot(1, 2, 2)
            cmap = 'gray' if imagen_aplicada.ndim == 2 else None
            plt.imshow(imagen_aplicada, cmap=cmap)
            plt.title("Imagen aislada (bebé)")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

        return mascara_objeto, imagen_aplicada

    except Exception as e:
        print(f"❌ Error al extraer el objeto principal:\n{e}")
        return None, None