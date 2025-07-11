from skimage.color import gray2rgb
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import img_as_ubyte

def dibujar_contornos_globulos(imagen_original, etiquetas_rojos, mascara_borde_rojos, mascara_blancos):
    """
    Dibuja los contornos de glóbulos detectados:
    - Glóbulos rojos internos → a partir de etiquetas (watershed)
    - Glóbulos rojos en borde → a partir de máscara binaria
    - Glóbulos blancos → a partir de máscara binaria
    
    Colores:
    🔴 Glóbulos rojos: rojo
    ⚪ Glóbulos blancos: azul

    :param imagen_original: Imagen RGB original
    :param etiquetas_rojos: Etiquetas de glóbulos rojos (resultado del watershed)
    :param mascara_borde_rojos: Máscara binaria de glóbulos rojos en los bordes
    :param mascara_blancos: Máscara binaria de glóbulos blancos
    :return: Imagen RGB con contornos superpuestos
    """
    # Asegurar formato RGB
    if len(imagen_original.shape) == 2:
        imagen_rgb = gray2rgb(imagen_original)
    else:
        imagen_rgb = imagen_original.copy()

    # Pasar a BGR para dibujar con OpenCV
    imagen_bgr = cv2.cvtColor(imagen_rgb, cv2.COLOR_RGB2BGR)

    # --- Dibujar contornos de etiquetas (rojos internos con watershed) ---
    for region in regionprops(etiquetas_rojos):
        # Coordenadas de cada objeto
        minr, minc, maxr, maxc = region.bbox
        # Máscara de la región actual
        mascara_region = (etiquetas_rojos == region.label).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imagen_bgr, contornos, -1, (0, 0, 255), 2)  # rojo (BGR)

    # --- Dibujar contornos de glóbulos rojos en el borde ---
    contornos_borde, _ = cv2.findContours(mascara_borde_rojos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen_bgr, contornos_borde, -1, (0, 0, 255), 2)  # rojo también

    # --- Dibujar contornos de glóbulos blancos ---
    contornos_blancos, _ = cv2.findContours(mascara_blancos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imagen_bgr, contornos_blancos, -1, (255, 0, 0), 2)  # azul

    # Convertir de nuevo a RGB
    imagen_final = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2RGB)
    return imagen_final
