import cv2
import numpy as np

def obtener_semilla_automatica(mascara_binaria):
    """
    Detecta la componente más grande en una máscara binaria y retorna su centroide como semilla.

    :param mascara_binaria: Imagen binaria (np.ndarray)
    :return: Tupla (fila, columna) con la semilla detectada automáticamente.
    """
    # Asegurar binaria
    _, binaria = cv2.threshold(mascara_binaria, 127, 255, cv2.THRESH_BINARY)

    # Etiquetado de componentes conectados
    num_labels, labels = cv2.connectedComponents(binaria)

    max_area = 0
    semilla = None

    for label in range(1, num_labels):  # Ignorar fondo (0)
        mascara_componente = (labels == label).astype(np.uint8)
        area = cv2.countNonZero(mascara_componente)

        if area > max_area:
            max_area = area
            M = cv2.moments(mascara_componente)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                semilla = (cy, cx)  # (fila, columna)

    return semilla
