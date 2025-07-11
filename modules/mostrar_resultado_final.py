import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.color import gray2rgb
from matplotlib.patches import Patch

def mostrar_resultado_final(imagen_original, etiquetas_rojos, mascara_borde_rojos, mascara_blancos):
    """
    Muestra la imagen original con bordes de glóbulos rojos (incluyendo los de bordes) y blancos superpuestos.
    Incluye leyenda clara debajo de la imagen.

    :param imagen_original: Imagen original (RGB o escala de grises)
    :param etiquetas_rojos: Etiquetas de glóbulos rojos internos (watershed)
    :param mascara_borde_rojos: Máscara binaria de glóbulos rojos en borde
    :param mascara_blancos: Máscara binaria de glóbulos blancos completos
    """
    # Convertir a RGB si es imagen en escala de grises
    if imagen_original.ndim == 2:
        imagen = gray2rgb(imagen_original)
    else:
        imagen = imagen_original.copy()

    resultado = imagen.copy()

    # 🔴 Dibujar contornos de glóbulos rojos internos
    for label in np.unique(etiquetas_rojos):
        if label == 0:
            continue
        mascara = (etiquetas_rojos == label).astype(np.uint8)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(resultado, contornos, -1, (255, 0, 0), 2)  # rojo

    # 🔴 Dibujar contornos de glóbulos rojos en borde (también en rojo)
    contornos_borde, _ = cv2.findContours(mascara_borde_rojos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultado, contornos_borde, -1, (255, 0, 0), 2)  # rojo

    # 🔵 Dibujar contornos de glóbulos blancos
    contornos_blancos, _ = cv2.findContours(mascara_blancos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resultado, contornos_blancos, -1, (0, 102, 255), 2)  # azul

    # 📌 Mostrar resultado
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(resultado)
    ax.axis("off")
    ax.set_title("Resultado final: glóbulos detectados")

    # 📌 Agregar leyenda debajo
    leyenda = [
        Patch(color=(1, 0, 0), label='Glóbulos rojos'),
        Patch(color=(0, 0.4, 1), label='Glóbulos blancos')
    ]
    plt.legend(
        handles=leyenda,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=False
    )

    plt.tight_layout()
    plt.show()