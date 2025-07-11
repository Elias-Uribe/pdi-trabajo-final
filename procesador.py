from modules.umbralizar import umbralizar_por_rango_opencv
from modules.leer_imagen_estandarizada import leer_imagen_estandarizada
from modules.obtener_banda_lab import obtener_banda_lab
from modules.contar_objetos_en_mascara import contar_objetos_en_mascara
from modules.dilatar_mascara import dilatar_mascara
from modules.restar_mascaras_binarias import restar_mascaras_binarias
from modules.erosionar_mascara import erosionar_mascara
from modules.segmentar_globulos_blancos_completos import segmentar_globulos_blancos_completos
from modules.eliminar_objetos_pequeños import eliminar_objetos_pequeños
from modules.separar_globulos_watershed import separar_globulos_watershed
from modules.contar_globulos_en_etiquetas import contar_globulos_en_etiquetas
from modules.cerrar_huecos_morfologicos import cerrar_huecos_morfologicos
from modules.aplicar_mediana_repetida import aplicar_mediana_repetida
from modules.separar_globulos_en_borde import separar_globulos_en_borde
from modules.mostrar_resultado_final import mostrar_resultado_final
from modules.umbral_adaptativo import umbral_adaptativo

""" 
PROCESAMIENTO DIGITAL DE IMÁGENES 1
CURSADA PRIMER CUATRIMESTRE 2025
TRABAJO FINAL

PROBLEMA:
Conteo de componentes en análisis de sangre.

OBJETIVO GENERAL:
Determinar la cantidad de glóbulos rojos y blancos en imágenes digitales de microscopía óptica.

OBJETIVOS ESPECÍFICOS:
1_ Establecer y definir digitalmente los patrones visuales que permiten identificar glóbulos rojos y blancos.
2_ Determinar en una imagen la cantidad de glóbulos rojos.
3_ Determinar la existencia de glóbulos blancos.
4_ Si existen glóbulos blancos, analizar si corresponde a un linfocito, en base a su núcleo.

---

🔍 1. Definición digital de patrones visuales

Se identificaron características distintivas para poder separar los glóbulos rojos y blancos:

🩸 Glóbulos Rojos:
- Forma: circular, centro más claro, contorno marcado (aspecto de dona).
- Textura: uniforme.
- Color: rosado en RGB, bien contrastado en el canal **L de LAB** (luminancia).

⚪ Glóbulos Blancos:
- Tamaño mayor.
- Forma irregular, núcleo morado visible.
- Textura interna.
- El **canal B del espacio LAB** permite resaltar el núcleo, facilitando su detección sin confundirlo con plaquetas.

---

⚙️ 2. Estrategia de procesamiento

El siguiente flujo de trabajo fue implementado para detectar y contar correctamente los glóbulos:

**Paso 1: Lectura de la imagen y conversión a LAB**
- Se convierte la imagen a espacio de color LAB.
- Se extraen dos bandas:
  - Canal L (luminancia) para glóbulos rojos.
  - Canal B para glóbulos blancos.

**Paso 2: Suavizado (filtro de mediana aplicado múltiples veces)**
- Se utiliza el filtro de mediana para eliminar ruido sin perder bordes.
- Se aplica 3 veces sobre glóbulos rojos y 2 sobre blancos.

**Paso 3: Umbralización por rango (threshold por canal)**
- Se define un rango para obtener una máscara binaria para cada tipo celular.
- En el caso de los glóbulos rojos, se invierte la máscara.

**Paso 4: Segmentación de glóbulos blancos completos**
- A partir de su núcleo, se expande la región para incluir el citoplasma.
- Se utiliza tolerancia de intensidad para evitar incluir células vecinas.

**Paso 5: Cierre morfológico + dilatación**
- Se cierran huecos internos del glóbulo blanco.
- Se dilata la máscara para asegurar cobertura completa.

**Paso 6: Eliminación del glóbulo blanco de la máscara de rojos**
- Se hace una resta entre la máscara de glóbulos rojos y blancos.

**Paso 7: Watershed para separar glóbulos rojos pegados**
- Se aplica transformada de distancia y detección de máximos.
- Se corrigen objetos no detectados en los bordes.

**Paso 8: Postprocesamiento**
- Se eliminan componentes pequeños que no son glóbulos.
- Se cuenta el total de etiquetas usando `cv2.connectedComponents` y se corrigen bordes.

---

✅ Resultado final:
Se obtienen:
- Una máscara con glóbulos rojos limpios y separados.
- Una máscara para glóbulos blancos con su citoplasma completo.
- Conteo para ambos tipos de células.
"""
def procesar_imagen(ruta_imagen): 
    # Paso 1: Leo la imagen
    imagen = leer_imagen_estandarizada(ruta_imagen)

    # Paso 2: Obtengo las bandas que resaltan mejor lo que busco
    banda_globulos_rojos = obtener_banda_lab(imagen, 'L')  # Luminancia para glóbulos rojos
    banda_globulos_blancos = obtener_banda_lab(imagen, 'B')  # Canal B para núcleo de glóbulos blancos

    # Paso 3: Aplico el filtro de la mediana para reducir el ruido y nivelar un poco los colores
    img_glogulos_rojos_suavizada = aplicar_mediana_repetida(banda_globulos_rojos, radio=6, repeticiones=3)
    img_glogulos_blancos_suavizada = aplicar_mediana_repetida(banda_globulos_blancos, radio=6, repeticiones=1)

    # Paso 4: Obtengo las respectivas mascaras con un umbral por un rango específico para cada globulo 
    mascara_binaria_globulos_rojos = umbralizar_por_rango_opencv(img_glogulos_rojos_suavizada, 146, 255, invertir=True)
    mascara_binaria_globulos_blancos = umbralizar_por_rango_opencv(img_glogulos_blancos_suavizada, 0, 104)

    # Paso 5: Segmento los glóbulos blancos completos (núcleo + citoplasma)
    mascara_globulos_blancos_completos = segmentar_globulos_blancos_completos(
        img_glogulos_blancos_suavizada,
        mascara_binaria_globulos_blancos,
        tolerancia=18,
        mostrar=True
    )

    # Paso 6: Erosionar la máscara para eliminar pequeñas irregularidades
    mascara_erosionada = erosionar_mascara(mascara_binaria_globulos_rojos, kernel_size=3, iteraciones=4)

    # Paso 7: Dilalar mascara de los glóbulos blancos para mejorar la segmentación
    mascara_globulos_blancos_dilatada = dilatar_mascara(mascara_globulos_blancos_completos, kernel_size=3, iteraciones=10)

    # Paso 8: Cierre morfológico para evitar huecos en glóbulos blancos
    mascara_globulos_blancos_cerrada = cerrar_huecos_morfologicos(mascara_globulos_blancos_dilatada, kernel_size=51, mostrar=True)

    # Paso 9: Quito los glóbulos blancos de la máscara de glóbulos rojos (resto las mascaras)
    mascara_globulos_rojos_sin_blanco = restar_mascaras_binarias(mascara_erosionada, mascara_globulos_blancos_cerrada)

    # Paso 10: Elimino objetos pequeños que no son glóbulos rojos
    mascara_globulos_rojos_limpia = eliminar_objetos_pequeños(mascara_globulos_rojos_sin_blanco, min_pixeles=80)

    # Paso 11: Separo los glóbulos en borde y en interior, los glòbulos del borde no serán procesados por watershed
    mascara_interior, mascara_borde = separar_globulos_en_borde(mascara_globulos_rojos_limpia, mostrar=True)

    # Paso 12:  Aplico watershed a glóbulos internos para separar pegados
    etiquetas_interior = separar_globulos_watershed(mascara_interior, min_pixeles=80, min_distance=10, mostrar=True)

    # Paso 13: Conteo total de glóbulos rojos
    cantidad_interior = contar_globulos_en_etiquetas(etiquetas_interior)
    cantidad_borde = contar_objetos_en_mascara(mascara_borde)
    cantidad_globulos_rojos = cantidad_interior + cantidad_borde

    # Paso 14: Conteo de glóbulos blancos
    cantidad_globulos_blancos = contar_objetos_en_mascara(mascara_globulos_blancos_cerrada)

    mascara_globulos_blancos_contorno = cerrar_huecos_morfologicos(mascara_globulos_blancos_completos, kernel_size=51, mostrar=True)

    # Mostrar imagen final
    mostrar_resultado_final(
        imagen_original=imagen, 
        etiquetas_rojos=etiquetas_interior,
        mascara_borde_rojos=mascara_borde,
        mascara_blancos=mascara_globulos_blancos_contorno
    )

    return cantidad_globulos_rojos, cantidad_globulos_blancos
