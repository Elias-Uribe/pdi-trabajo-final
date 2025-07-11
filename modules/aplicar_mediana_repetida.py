from skimage.filters import median
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage import io
import numpy as np

def aplicar_mediana_repetida(imagen_entrada, radio=3, repeticiones=1):
    """
    Aplica el filtro de mediana varias veces sobre una imagen.

    :param imagen_entrada: Ruta o array NumPy (RGB o escala de grises)
    :param radio: Radio del disco estructurante (tamaño del filtro)
    :param repeticiones: Cantidad de veces que se aplica la mediana
    :return: Imagen suavizada
    """
    try:
        # Leer imagen si es ruta
        if isinstance(imagen_entrada, str):
            imagen = io.imread(imagen_entrada)
        else:
            imagen = imagen_entrada

        # Si es RGB, convertir a escala de grises
        if imagen.ndim == 3:
            imagen = rgb2gray(imagen)

        # Convertir a uint8 si hace falta
        imagen = img_as_ubyte(imagen)

        # Aplicar mediana varias veces
        for _ in range(repeticiones):
            imagen = median(imagen, disk(radio))

        return imagen

    except Exception as e:
        print(f"❌ Error al aplicar mediana repetida: {e}")
        return None
