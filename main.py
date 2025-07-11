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
from modules.datos_estadisticos import datos_estadisticos
from modules.umbral_adaptativo import umbral_adaptativo
from modules.umbral_otsu import umbral_otsu
from modules.separar_felzenszwalb import separar_felzenszwalb
from modules.cerrar_huecos_precisos import cerrar_huecos_precisos
from modules.cerrar_huecos_con_grietas import cerrar_huecos_con_grietas
from modules.cerrar_huecos_con_contornos import cerrar_huecos_con_contornos
from modules.cerrar_huecos_inteligente import cerrar_huecos_inteligente
from modules.guardar_mascaras_sanguineas import guardar_mascaras_sanguineas
from modules.procesar_carpeta_de_imagenes import procesar_carpeta_de_imagenes
import math

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

Paso 7: Limpieza de objetos no relevantes
- Se eliminan objetos demasiado pequeños para ser glóbulos reales.

Paso 8: Separación de glóbulos en bordes vs interior
- Se detectan los glóbulos que tocan el borde de la imagen.
- Se separan para evitar errores de segmentación.

Paso 9: Watershed sobre glóbulos rojos internos
- Se aplica transformada de distancia y detección de máximos locales.
- Watershed permite separar células que están pegadas.
- Solo se aplica a glóbulos internos para evitar errores en los bordes.

Paso 10: Conteo y visualización final
- Se cuenta por separado:
    Glóbulos rojos internos (por watershed).
    Glóbulos en los bordes.
    Glóbulos blancos.

- Se genera una visualización final con bordes de color para cada tipo:
    🔴 Rojo: glóbulos rojos.
    🔵 Azul: glóbulos blancos.
---

✅ Resultado Final:
- Glóbulos rojos separados, incluso si estaban pegados.
- Glóbulos blancos completos, sin incluir núcleos aislados ni ruido.
- Imagen final con bordes coloreados y conteo preciso por tipo.
"""

""" # Rutas de imágenes
ruta_local = "Tareas/Proyecto final - Elias Uribe/images/image-83.png"

# Paso 1: Leo la imagen
imagen = leer_imagen_estandarizada(ruta_local)

# Paso 2: Obtengo las bandas que resaltan mejor lo que busco
banda_globulos_rojos = obtener_banda_lab(imagen, 'L')  # Luminancia para glóbulos rojos
banda_globulos_blancos = obtener_banda_lab(imagen, 'B')  # Canal B para núcleo de glóbulos blancos

# Paso 3: Aplico el filtro de la mediana para reducir el ruido y nivelar un poco los colores
img_glogulos_rojos_suavizada = aplicar_mediana_repetida(banda_globulos_rojos, radio=6, repeticiones=3)
img_glogulos_blancos_suavizada = aplicar_mediana_repetida(banda_globulos_blancos, radio=6, repeticiones=1)

media_L, std_L = datos_estadisticos(img_glogulos_rojos_suavizada)
media_B, std_B = datos_estadisticos(img_glogulos_blancos_suavizada)
 """
""" # Paso 4: Obtengo las respectivas mascaras con un umbral por un rango específico para cada globulo 
mascara_binaria_globulos_rojos = umbralizar_por_rango_opencv(img_glogulos_rojos_suavizada, math.trunc(media_L - std_L), math.trunc(media_L + std_L), invertir=True)
mascara_binaria_globulos_blancos = umbralizar_por_rango_opencv(img_glogulos_blancos_suavizada, umbral_min=0, umbral_max=135)

umbral_adapt_globulo_rojo = umbral_adaptativo(img_glogulos_rojos_suavizada, bloque=15, C=3)
umbral_adapt_globulo_blanco = umbral_adaptativo(img_glogulos_blancos_suavizada, bloque=61, C=4)
 """
""" mascara_rojos, umbral_globulo_rojo = umbral_otsu(img_glogulos_rojos_suavizada, invertir=True)
mascara_blancos, umbral_globulo_blanco = umbral_otsu(img_glogulos_blancos_suavizada, invertir=True)

# Paso 5: Segmento los glóbulos blancos completos (núcleo + citoplasma)
mascara_globulos_blancos_completos = segmentar_globulos_blancos_completos(
    img_glogulos_blancos_suavizada,
    mascara_blancos,
    tolerancia=std_B+(std_B/2),
    mostrar=True
)

# Paso 6: Erosionar la máscara para eliminar pequeñas irregularidades
mascara_erosionada = erosionar_mascara(mascara_rojos, kernel_size=3, iteraciones=4)

# Paso 7: Dilalar mascara de los glóbulos blancos para mejorar la segmentación
mascara_globulos_blancos_dilatada = dilatar_mascara(mascara_globulos_blancos_completos, kernel_size=3, iteraciones=10)

# Paso 8: Cierre morfológico para evitar huecos en glóbulos blancos
mascara_globulos_blancos_cerrada = cerrar_huecos_morfologicos(mascara_globulos_blancos_dilatada, kernel_size=51, mostrar=True)

# Paso 9: Quito los glóbulos blancos de la máscara de glóbulos rojos (resto las mascaras)
mascara_globulos_rojos_sin_blanco = restar_mascaras_binarias(mascara_erosionada, mascara_globulos_blancos_cerrada)

# Paso 10: Elimino objetos pequeños que no son glóbulos rojos
mascara_globulos_rojos_limpia = eliminar_objetos_pequeños(mascara_globulos_rojos_sin_blanco, min_pixeles=80)

mascara_globulos_rojos_cerrada = cerrar_huecos_precisos(mascara_globulos_rojos_limpia, mostrar=True)
#mascara_globulos_rojos_cerrada = cerrar_huecos_con_grietas(mascara_globulos_rojos_limpia, kernel_size=23)
# No funciona bien con grietas, pero si con huecos internos, ya que deberia dilar las celulas, pero el tema es de dependiendo
# el valor del kernel_size, se agranda la celula, y se termina uniendo con otras

#mascara_globulos_rojos_cerrada = cerrar_huecos_inteligente(mascara_globulos_rojos_limpia, erosion_iter=65)
# Funciona bien, pero no me soluciona el tema de los huecos que no estan totalmente rodeados por el borde de la celula
# para esto me quedo con la funcion cerrar_huecos_precisos """

""" guardar_mascaras_sanguineas(
    mascara_rojos=mascara_globulos_rojos_cerrada,
    mascara_blancos=mascara_globulos_blancos_cerrada,
    nombre_base="4",
    carpeta_destino="data/masks/"
)
 """
""" # Paso 11: Separo los glóbulos en borde y en interior, los glòbulos del borde no serán procesados por watershed
mascara_interior, mascara_borde = separar_globulos_en_borde(mascara_globulos_rojos_cerrada, mostrar=True)

# Paso 12:  Aplico watershed a glóbulos internos para separar pegados
etiquetas_interior = separar_globulos_watershed(mascara_interior, min_pixeles=80, min_distance=10, mostrar=True)
 """
""" Probar alguna otro alterneativo a watershed, como por ejemplo: Quickshift, MeanShift, Felzenszwalb, etc. """
#etiquetas_interior = separar_felzenszwalb(mascara_interior, mostrar=True)

# Paso 13: Conteo total de glóbulos rojos
""" cantidad_interior = contar_globulos_en_etiquetas(etiquetas_interior)
cantidad_borde = contar_objetos_en_mascara(mascara_borde)
cantidad_globulos_rojos = cantidad_interior + cantidad_borde
print(f"🔴 Glóbulos rojos detectados: {cantidad_globulos_rojos}")

# Paso 14: Conteo de glóbulos blancos
cantidad_globulos_blancos = contar_objetos_en_mascara(mascara_globulos_blancos_cerrada)
print(f"⚪ Glóbulos blancos detectados: {cantidad_globulos_blancos}")

mascara_globulos_blancos_contorno = cerrar_huecos_morfologicos(mascara_globulos_blancos_completos, kernel_size=51, mostrar=True)

# Mostrar imagen final
mostrar_resultado_final(
    imagen_original=imagen,
    etiquetas_rojos=etiquetas_interior,
    mascara_borde_rojos=mascara_borde,
    mascara_blancos=mascara_globulos_blancos_contorno
) """



procesar_carpeta_de_imagenes(
    carpeta_entrada="Tareas\Proyecto final - Elias Uribe\images\dataset-master\dataset-master\JPEGImages",
    carpeta_salida="data/masks/"
)
