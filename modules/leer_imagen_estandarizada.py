from skimage import io
import numpy as np
from skimage.color import rgb2gray

def leer_imagen_estandarizada(imagen_entrada, forzar_gris=False):
    """
    Acepta una ruta o una imagen en array. Si es RGB y se indica, la convierte a escala de grises.

    :param imagen_entrada: Ruta (str) o imagen NumPy
    :param forzar_gris: Si True, convierte a escala de grises
    :return: Imagen como array NumPy, uint8 o float según formato original
    """
    try:
        # Leer si es ruta
        if isinstance(imagen_entrada, str):
            imagen = io.imread(imagen_entrada)
        elif isinstance(imagen_entrada, np.ndarray):
            imagen = imagen_entrada.copy()
        else:
            raise TypeError("La entrada debe ser una ruta (str) o un array NumPy.")

        # Convertir a escala de grises si se indica
        if forzar_gris and imagen.ndim == 3:
            imagen = rgb2gray(imagen)
        
        return imagen

    except Exception as e:
        print(f"❌ Error al leer la imagen: {e}")
        return None
