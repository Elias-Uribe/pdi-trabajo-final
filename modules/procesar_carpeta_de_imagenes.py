import os
import cv2
import math
from tqdm import tqdm  # Para barra de progreso (opcional)
import numpy as np

from modules.leer_imagen_estandarizada import leer_imagen_estandarizada
from modules.obtener_banda_lab import obtener_banda_lab
from modules.dilatar_mascara import dilatar_mascara
from modules.restar_mascaras_binarias import restar_mascaras_binarias
from modules.erosionar_mascara import erosionar_mascara
from modules.segmentar_globulos_blancos_completos import segmentar_globulos_blancos_completos
from modules.eliminar_objetos_pequeños import eliminar_objetos_pequeños
from modules.cerrar_huecos_morfologicos import cerrar_huecos_morfologicos
from modules.aplicar_mediana_repetida import aplicar_mediana_repetida
from modules.datos_estadisticos import datos_estadisticos
from modules.umbral_otsu import umbral_otsu
from modules.cerrar_huecos_precisos import cerrar_huecos_precisos
from modules.guardar_mascaras_sanguineas import guardar_mascaras_sanguineas

def procesar_carpeta_de_imagenes(carpeta_entrada, carpeta_salida):
    """
    Procesa todas las imágenes en una carpeta y guarda sus máscaras de glóbulos rojos y blancos.
    
    :param carpeta_entrada: Ruta donde están las imágenes originales.
    :param carpeta_salida: Ruta donde se guardarán las máscaras.
    """
    # Obtener lista de imágenes
    rutas_imagenes = [os.path.join(carpeta_entrada, f)
                      for f in os.listdir(carpeta_entrada)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Asegurar que carpeta de salida exista
    os.makedirs(carpeta_salida, exist_ok=True)

    # Procesar cada imagen
    for ruta in tqdm(rutas_imagenes, desc="Procesando imágenes"):
        nombre_base = os.path.splitext(os.path.basename(ruta))[0]

        # Paso 1: Leer imagen
        imagen = leer_imagen_estandarizada(ruta)

        # Paso 2: Extraer bandas
        banda_globulos_rojos = obtener_banda_lab(imagen, 'L')
        banda_globulos_blancos = obtener_banda_lab(imagen, 'B')

        # Paso 3: Suavizado
        img_rojos = aplicar_mediana_repetida(banda_globulos_rojos, radio=6, repeticiones=3)
        img_blancos = aplicar_mediana_repetida(banda_globulos_blancos, radio=6, repeticiones=1)

        # Paso 4: Estadísticas + Umbralización (Otsu)
        media_L, std_L = datos_estadisticos(img_rojos)
        media_B, std_B = datos_estadisticos(img_blancos)

        mascara_rojos, _ = umbral_otsu(img_rojos, invertir=True, mostrar=False)
        mascara_blancos, _ = umbral_otsu(img_blancos, invertir=True, mostrar=False)

        # Paso 5: Crecimiento región glóbulos blancos
        mascara_blancos_completa = segmentar_globulos_blancos_completos(
            img_blancos,
            mascara_blancos,
            tolerancia=std_B+(std_B/2),
            mostrar=False
        )

        # Paso 6: Operaciones morfológicas
        erosionada = erosionar_mascara(mascara_rojos, kernel_size=3, iteraciones=4, mostrar=False)
        dilatada_blanco = dilatar_mascara(mascara_blancos_completa, kernel_size=3, iteraciones=10, mostrar=False)
        cerrada_blanco = cerrar_huecos_morfologicos(dilatada_blanco, kernel_size=51, mostrar=False)

        # Paso 7: Restar glóbulos blancos
        sin_blanco = restar_mascaras_binarias(erosionada, cerrada_blanco, mostrar=False)

        # Paso 8: Limpiar
        limpios = eliminar_objetos_pequeños(sin_blanco, min_pixeles=80, mostrar=False)
        cerrada_rojo = cerrar_huecos_precisos(limpios, mostrar=False)

        # Paso 9: Guardar las máscaras
        guardar_mascaras_sanguineas(
            mascara_rojos=cerrada_rojo,
            mascara_blancos=cerrada_blanco,
            nombre_base=nombre_base,
            carpeta_destino=carpeta_salida
        )
