import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lista de imágenes a procesar
imagenes = ['figures.jpg', 'blancoNegro.png', 'figuraV.png', 'ninioB.png', 'riceB.png']

# Ecualizar
def ecualizar_histograma(imagen):
    # Calcula el histograma
    histograma = np.zeros(256)
    for fila in range(imagen.shape[0]):
        for columna in range(imagen.shape[1]):
            histograma[imagen[fila, columna]] += 1
    
    # Calcula la distribución acumulada (CDF)
    distribucion_acumulada = np.zeros(256)
    distribucion_acumulada[0] = histograma[0]
    for indice in range(1, 256):
        distribucion_acumulada[indice] = distribucion_acumulada[indice-1] + histograma[indice]
    
    # Normaliza la CDF: (CDF - min(CDF)) * 255 / (total_pixeles - min(CDF))
    # Esto mapea los valores de intensidad a un rango completo de 0 a 255
    minimo_distribucion = np.min(distribucion_acumulada[np.nonzero(distribucion_acumulada)])
    total_pixeles = imagen.shape[0] * imagen.shape[1]
    distribucion_normalizada = ((distribucion_acumulada - minimo_distribucion) * 255 / (total_pixeles - minimo_distribucion)).astype(np.uint8)
    
    # Aplica la ecualización
    imagen_resultante = np.zeros_like(imagen)
    for fila in range(imagen.shape[0]):
        for columna in range(imagen.shape[1]):
            imagen_resultante[fila, columna] = distribucion_normalizada[imagen[fila, columna]]
    
    return imagen_resultante, histograma

# Procesa cada imagen
for nombre_imagen in imagenes:
    # Lee la imagen en escala de grises
    imagen_original = cv2.imread(nombre_imagen, cv2.IMREAD_GRAYSCALE)
    
    # Verifica
    if imagen_original is None:
        print(f"No se pudo cargar '{nombre_imagen}'. Verifica el archivo.")
        continue
    
    # Calcula histograma inicial
    histograma_inicial = np.zeros(256)
    for fila in range(imagen_original.shape[0]):
        for columna in range(imagen_original.shape[1]):
            histograma_inicial[imagen_original[fila, columna]] += 1
    
    # Ecualiza la imagen
    imagen_ecualizada, _ = ecualizar_histograma(imagen_original)
    
    # Calcula histograma ecualizado
    histograma_final = np.zeros(256)
    for fila in range(imagen_ecualizada.shape[0]):
        for columna in range(imagen_ecualizada.shape[1]):
            histograma_final[imagen_ecualizada[fila, columna]] += 1
    
    # Crea y muestra la figuras
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.bar(range(256), histograma_inicial)
    plt.title('Histograma Inicial')
    plt.xlabel('Nivel de Gris')
    plt.ylabel('Frecuencia')
    
    plt.subplot(2, 2, 2)
    plt.imshow(imagen_original, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.bar(range(256), histograma_final)
    plt.title('Histograma Ecualizado')
    plt.xlabel('Nivel de Gris')
    plt.ylabel('Frecuencia')
    
    plt.subplot(2, 2, 4)
    plt.imshow(imagen_ecualizada, cmap='gray')
    plt.title('Imagen Ecualizada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    plt.pause(0.1)

print("Procesamiento completado.")
