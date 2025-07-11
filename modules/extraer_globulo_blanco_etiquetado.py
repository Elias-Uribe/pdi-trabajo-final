import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

def extraer_globulo_blanco_etiquetado(mascara_total, mascara_globulo_blanco):
    """
    Extrae el glóbulo blanco completo usando etiquetado y la intersección con su máscara.

    :param mascara_total: Máscara binaria con todos los glóbulos.
    :param mascara_globulo_blanco: Máscara binaria del glóbulo blanco (dilatada).
    :return: 
        - glóbulo_blanco_mascara: imagen binaria solo con el glóbulo blanco completo.
        - etiqueta_encontrada: número de la etiqueta encontrada (para depuración o análisis).
    """
    try:
        # Etiquetar objetos en la máscara total
        etiquetas = label(mascara_total)

        # Intersección entre las etiquetas y la máscara del glóbulo blanco
        interseccion = etiquetas * (mascara_globulo_blanco > 0)

        # Obtener la etiqueta del glóbulo blanco (más frecuente en la intersección)
        etiquetas_presentes = interseccion[interseccion > 0]
        if len(etiquetas_presentes) == 0:
            print("❌ No se encontró intersección entre las máscaras.")
            return None, None

        etiqueta_blanco = np.bincount(etiquetas_presentes).argmax()

        # Crear una nueva máscara binaria solo con esa etiqueta
        globulo_blanco_mascara = (etiquetas == etiqueta_blanco).astype(np.uint8) * 255

        # Mostrar resultado
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(mascara_total, cmap='gray')
        plt.title("Máscara Total (rojos + blancos)")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(mascara_globulo_blanco, cmap='gray')
        plt.title("Máscara del Glóbulo Blanco (dilatada)")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(globulo_blanco_mascara, cmap='gray')
        plt.title(f"Glóbulo Blanco Extraído (etiqueta {etiqueta_blanco})")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return globulo_blanco_mascara, etiqueta_blanco

    except Exception as e:
        print(f"❌ Error al extraer glóbulo blanco etiquetado: {e}")
        return None, None
