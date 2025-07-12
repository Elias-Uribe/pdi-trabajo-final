# 🩸 Proyecto Final - Procesamiento Digital de Imágenes I

**Nombre:** Elías Uribe  
**Materia:** Procesamiento Digital de Imágenes I  
**Año:** 2025  
**Trabajo Final:** Conteo de glóbulos rojos y blancos en imágenes de análisis de sangre

---

## 🧪 Descripción

## Este proyecto detecta, segmenta y cuenta glóbulos rojos y blancos a partir de imágenes microscópicas de sangre. Se utilizan técnicas de procesamiento morfológico clásico, junto con modelos de segmentación profunda como U-Net, para lograr una segmentación precisa y adaptable.

## 📁 Estructura del Proyecto

```
pdi-trabajo-final
│
├── images/               # Imágenes de entrada (microscopía)
├── masks/                # Máscaras generadas para entrenamiento del modelo
├── data/
│   ├── images/           # Imágenes para entrenamiento U-Net
│   └── masks/            # Máscaras binarias de glóbulos blancos
│
├── modules/              # Módulos reutilizables y funciones específicas
│
├── interface.py          # ⚙️ Ejecutable principal para conteo tradicional
├── procesador.py         # Contiene la lógica principal del procesamiento
│
├── unet_model.py         # Entrenamiento de modelo U-Net (usa /data/images y /data/masks)
├── pruebas_model.py      # Utiliza un modelo .keras entrenado para segmentar glóbulos blancos
│
├── mejor_modelo.keras    # 🧠 Modelo entrenado con U-Net
└── README.md             # Documentación
```

---

## ▶️ ¿Cómo Ejecutar?

> ✅ Requisitos: Python 3.8+ y los siguientes paquetes:

- `opencv-python`
- `numpy`
- `matplotlib`
- `scikit-image`
- `scipy`

### 1. Instalar dependencias (si no las tenés)

```bash
pip install opencv-python numpy matplotlib scikit-image scipy
```

### 2. Ejecutar el proyecto desde consola

```bash
python interface.py
```

El script procesará la imagen seleccionada desde su computadora, mostrará visualmente los resultados y contará los glóbulos rojos y blancos detectados.

---

## 🧠 Entrenar el modelo U-Net

---

Asegurate de tener:

- Carpeta data/images/: imágenes reales en color
- Carpeta data/masks/: máscaras binarias de glóbulos blancos (fondo = negro, glóbulos = blanco)

Instalar dependencias adicionales:

- pip install tensorflow keras scikit-learn

Ejecutar el script de entrenamiento:

- python unet_model.py

---

## 🛠️ Tecnologías y Técnicas Utilizadas

- Conversión de color (espacio LAB)
- Filtro de mediana
- Umbralización por rango (OpenCV)
- Operaciones morfológicas: erosión, dilatación, cierre
- Transformada de distancia + Watershed
- Etiquetado y conteo de componentes conectados
- Visualización final con bordes y leyenda

---

## 📌 Nota

Este proyecto fue desarrollado como trabajo final para la materia **Procesamiento Digital de Imágenes I** de la carrera de Ingeniería Informática.

---
