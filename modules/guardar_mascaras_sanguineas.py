import os
import cv2
import numpy as np

def guardar_mascaras_sanguineas(
    mascara_rojos: np.ndarray,
    mascara_blancos: np.ndarray,
    nombre_base: str = "mascara",
    carpeta_destino: str = "outputs"
):
    """
    Guarda las máscaras binarias de glóbulos rojos y blancos como imágenes PNG.

    :param mascara_rojos: np.ndarray - máscara binaria (0-255) de glóbulos rojos.
    :param mascara_blancos: np.ndarray - máscara binaria (0-255) de glóbulos blancos.
    :param nombre_base: nombre base de los archivos guardados (sin extensión).
    :param carpeta_destino: carpeta donde se guardarán las imágenes.
    """
    # Crear carpeta si no existe
    os.makedirs(carpeta_destino, exist_ok=True)

    # Rutas
    ruta_rojos = os.path.join(carpeta_destino, f"{nombre_base}_rojos.png")
    ruta_blancos = os.path.join(carpeta_destino, f"{nombre_base}_blancos.png")

    # Guardar imágenes (asegurarse que están en 8-bit para guardar correctamente)
    cv2.imwrite(ruta_rojos, mascara_rojos)
    cv2.imwrite(ruta_blancos, mascara_blancos)

    print(f"✅ Máscara de glóbulos rojos guardada en: {ruta_rojos}")
    print(f"✅ Máscara de glóbulos blancos guardada en: {ruta_blancos}")
