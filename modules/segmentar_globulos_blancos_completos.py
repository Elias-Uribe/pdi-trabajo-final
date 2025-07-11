import numpy as np
import cv2
import matplotlib.pyplot as plt
from modules.crecimiento_de_regiones import crecimiento_de_regiones

def segmentar_globulos_blancos_completos(imagen_gris, mascara_nucleos, tolerancia=8, mostrar=False):
    """
    Segmenta todos los glóbulos blancos (núcleo + citoplasma) usando crecimiento de regiones desde cada núcleo.

    :param imagen_gris: Imagen preprocesada (suavizada) en escala de grises donde se aplica crecimiento.
    :param mascara_nucleos: Máscara binaria que representa los núcleos de los glóbulos blancos.
    :param tolerancia: Tolerancia en la diferencia de intensidad para el crecimiento.
    :param mostrar: Si True, muestra la máscara resultante.
    :return: Máscara binaria con todos los glóbulos blancos completos (núcleo + citoplasma).
    """
    if imagen_gris.dtype != np.uint8:
        imagen_gris = (imagen_gris * 255).astype(np.uint8)

    # Etiquetado de componentes conectados en la máscara de núcleos
    num_labels, labels = cv2.connectedComponents(mascara_nucleos)

    # Crear máscara final vacía
    mascara_final = np.zeros_like(imagen_gris, dtype=np.uint8)

    for label in range(1, num_labels):  # saltar fondo
        # Crear máscara de la región actual
        mascara_componente = (labels == label).astype(np.uint8)

        # Obtener centroide de la región
        M = cv2.moments(mascara_componente)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        semilla = (cy, cx)

        # Aplicar crecimiento de regiones desde la semilla
        region_crecida = crecimiento_de_regiones(imagen_gris, semilla, tolerancia=tolerancia, mostrar=False)

        # Sumarla a la máscara final
        mascara_final = cv2.bitwise_or(mascara_final, region_crecida)

    if mostrar:
        plt.figure(figsize=(6, 5))
        plt.imshow(mascara_final, cmap='gray')
        plt.title("Máscara: glóbulos blancos completos")
        plt.axis('off')
        plt.show()

    return mascara_final
