import cv2
import numpy as np

def contar_objetos_en_mascara(mascara_binaria):
    """
    Cuenta los objetos en una máscara binaria utilizando componentes conectados.

    :param mascara_binaria: np.ndarray con valores 0 y 255.
    :return: cantidad de objetos encontrados (int)

    El etiquetado de componentes conectados es una técnica muy común en procesamiento de 
    imágenes para identificar y contar objetos en una imagen binaria (blanco y negro).

    Toma una imagen binaria donde los objetos son blancos (255) y el fondo es negro (0), y:
        - Busca grupos de píxeles blancos conectados entre sí.
        - A cada grupo lo asigna una "etiqueta" o número único (por ejemplo, 1, 2, 3, ...).
        - Genera una imagen de etiquetas, donde cada objeto tiene un número diferente.
        - Devuelve la cantidad total de objetos encontrados (sin contar el fondo).

    Tipos de conectividad:
        - 4-conectividad: píxeles vecinos por arriba, abajo, izquierda y derecha.
        - 8-conectividad: además de los anteriores, incluye las diagonales.

    Con cv2.connectedComponents, OpenCV usa 8-conectividad por defecto.

    ¿Para qué sirve?
        - Contar objetos (por ejemplo: células, monedas, glóbulos, etc.).
        - Distinguir cada objeto individual en una imagen.
        - Calcular propiedades de cada objeto (área, forma, posición...).
    """
    # Convertir la máscara a formato binario real (valores 0 y 1)
    mascara_binaria = (mascara_binaria > 0).astype(np.uint8)

    # Etiquetar componentes conectados
    cantidad_objetos, etiquetas = cv2.connectedComponents(mascara_binaria)

    # Restar 1 porque el label 0 es el fondo
    return cantidad_objetos - 1
