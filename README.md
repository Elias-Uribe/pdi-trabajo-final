# 🩸 Proyecto Final - Procesamiento Digital de Imágenes I

**Nombre:** Elías Uribe  
**Materia:** Procesamiento Digital de Imágenes I  
**Año:** 2025  
**Trabajo Final:** Conteo de glóbulos rojos y blancos en imágenes de análisis de sangre

---

## 🧪 Descripción

Este proyecto detecta, segmenta y cuenta glóbulos rojos y glóbulos blancos a partir de imágenes microscópicas digitales de sangre.  
Se utilizan técnicas de procesamiento morfológico, filtrado y segmentación (incluyendo watershed) para obtener resultados precisos.

---

## 📁 Estructura del Proyecto

```
Proyecto final - Elias Uribe/
│
├── images/         # Carpeta para imágenes de entrada (estas imágenes son de prueba, la propia interface te dejara seleccionar la imagen que quieras)
├── modules/        # Módulos reutilizables y funciones específicas
├── main.py         # Script auxiliar (no se ejecuta directamente, es para pruebas)
├── procesador.py   # Contiene la lógica principal del procesamiento
├── interface.py    # ⚙️ Ejecutable principal del proyecto
└── README.md       # Instrucciones de uso y documentación
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
