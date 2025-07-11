import cv2
import numpy as np
import matplotlib.pyplot as plt

def cerrar_huecos_con_contornos(mascara_binaria, mostrar=True):
    """
    Cierra huecos abiertos en glóbulos sin unir células, utilizando envolventes de contornos.

    :param mascara_binaria: Imagen binaria (0-255).
    :param mostrar: Mostrar comparación visual.
    :return: Nueva máscara binaria sin huecos.
    """
    # Asegurar binaria 0-255
    mascara_binaria = (mascara_binaria > 0).astype(np.uint8) * 255

    # Buscar contornos
    contornos, _ = cv2.findContours(mascara_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crear imagen vacía
    nueva_mascara = np.zeros_like(mascara_binaria)

    for contorno in contornos:
        # Opcional: usar cv2.convexHull(contorno) si querés convexidad estricta
        cv2.drawContours(nueva_mascara, [contorno], -1, 255, thickness=cv2.FILLED)

    if mostrar:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mascara_binaria, cmap='gray')
        plt.title("Original (con huecos)")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(nueva_mascara, cmap='gray')
        plt.title("Huecos cerrados por contorno")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return nueva_mascara
