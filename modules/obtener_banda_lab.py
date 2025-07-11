from skimage import io
import cv2

def obtener_banda_lab(imagen, canal):
    """
    Extrae una banda específica del espacio LAB: 'L', 'A' o 'B'.
    """
    if isinstance(imagen, str):
        imagen = io.imread(imagen)

    imagen_bgr = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2LAB)

    canal = canal.upper()
    canales = {"L": 0, "A": 1, "B": 2}

    if canal not in canales:
        raise ValueError("Canal inválido. Usar 'L', 'A' o 'B'.")

    return lab[:, :, canales[canal]]
