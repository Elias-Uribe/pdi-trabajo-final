import numpy as np

def contar_globulos_en_etiquetas(etiquetas):
    """
    Cuenta la cantidad de glóbulos (etiquetas únicas), ignorando fondo (0).
    """
    etiquetas_validas = etiquetas[etiquetas > 0]
    cantidad = len(np.unique(etiquetas_validas))
    return cantidad
