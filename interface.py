import tkinter as tk
from tkinter import filedialog, messagebox
from procesador import procesar_imagen

def seleccionar_imagen():
    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[("Archivos de imagen", "*.jpg *.png *.jpeg *.bmp")]
    )

    if ruta:
        try:
            # Procesar imagen con el pipeline definido
            rojos, blancos = procesar_imagen(ruta)

            # Mostrar resultados en ventana emergente
            messagebox.showinfo(
                "Resultados del análisis",
                f"🔴 Glóbulos rojos detectados: {rojos}\n⚪ Glóbulos blancos detectados: {blancos}"
            )
        except Exception as e:
            messagebox.showerror("Error en el procesamiento", str(e))

# Crear ventana principal
ventana = tk.Tk()
ventana.title("Conteo de globulos rojos y blancos - Análisis de Imágenes")
ventana.geometry("400x200")
ventana.resizable(False, False)

# Título
titulo = tk.Label(ventana, text="Conteo de globulos rojos y blancos", font=("Arial", 16))
titulo.pack(pady=20)

# Botón de selección
btn_cargar = tk.Button(
    ventana,
    text="Seleccionar imagen",
    command=seleccionar_imagen,
    font=("Arial", 12),
    bg="#4CAF50",
    fg="white",
    padx=10,
    pady=5
)
btn_cargar.pack()

# Ejecutar loop de la GUI
ventana.mainloop()
