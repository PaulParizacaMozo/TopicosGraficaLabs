import numpy as np
import matplotlib.pyplot as plt
import cv2

def calcularHistograma(imagenGris):
    histograma = np.zeros(256, dtype=int)
    filas, columnas = imagenGris.shape
    
    for i in range(filas):
        for j in range(columnas):
            valorPixel = imagenGris[i, j]
            histograma[valorPixel] += 1
    
    return histograma

def otsuGlobal(imagenGris):
    histograma = calcularHistograma(imagenGris)
    totalPixeles = imagenGris.size
    
    # probabilidades de cada nivel de gris
    probabilidades = np.zeros(256, dtype=float)
    for i in range(256):
        probabilidades[i] = histograma[i] / totalPixeles

    varianzaMaxima = 0
    umbralOptimo = 0

    for umbral in range(1, 256):
        # calculo de pesos y medias
        pesoFondo, mediaFondoNum = 0.0, 0.0
        for i in range(umbral):
            pesoFondo += probabilidades[i]
            mediaFondoNum += i * probabilidades[i]

        pesoObjeto, mediaObjetoNum = 0.0, 0.0
        for i in range(umbral, 256):
            pesoObjeto += probabilidades[i]
            mediaObjetoNum += i * probabilidades[i]

        if pesoFondo == 0 or pesoObjeto == 0:
            continue

        mediaFondo = mediaFondoNum / pesoFondo
        mediaObjeto = mediaObjetoNum / pesoObjeto
        
        # varianza entre clases
        varianzaEntreClases = pesoFondo * pesoObjeto * (mediaFondo - mediaObjeto) ** 2

        if varianzaEntreClases > varianzaMaxima:
            varianzaMaxima = varianzaEntreClases
            umbralOptimo = umbral

    # binarizacion
    filas, columnas = imagenGris.shape
    imagenBinarizada = np.zeros_like(imagenGris)
    for i in range(filas):
        for j in range(columnas):
            if imagenGris[i, j] > umbralOptimo:
                imagenBinarizada[i, j] = 255
    
    return imagenBinarizada, umbralOptimo

def otsuAdaptativo(imagenGris, anchoVentana, altoVentana):
    altura, ancho = imagenGris.shape
    imagenSegmentada = np.zeros_like(imagenGris)

    for fila in range(0, altura, altoVentana):
        for columna in range(0, ancho, anchoVentana):
            ventanaActual = imagenGris[fila:fila + altoVentana, columna:columna + anchoVentana]
            
            if ventanaActual.size == 0:
                continue

            # si la ventana es de un solo color, usa umbral medio
            if ventanaActual.min() == ventanaActual.max():
                umbralLocal = 127
            else:
                 _, umbralLocal = otsuGlobal(ventanaActual)

            # binariza la ventana local
            ventanaBinarizada = np.zeros_like(ventanaActual)
            vFilas, vCols = ventanaActual.shape
            for i in range(vFilas):
                for j in range(vCols):
                    if ventanaActual[i,j] > umbralLocal:
                        ventanaBinarizada[i,j] = 255
            
            imagenSegmentada[fila:fila + altoVentana, columna:columna + anchoVentana] = ventanaBinarizada
            
    return imagenSegmentada

def procesarYMostrarImagen(rutaImagen, anchoVentanaAdaptativo=50, altoVentanaAdaptativo=50):
    imagenGris = cv2.imread(rutaImagen, cv2.IMREAD_GRAYSCALE)

    # procsamiento de imgs
    imagenOtsu, umbralGlobal = otsuGlobal(imagenGris)
    imagenOtsuAdaptativo = otsuAdaptativo(imagenGris, 
                                          anchoVentana=anchoVentanaAdaptativo, 
                                          altoVentana=altoVentanaAdaptativo)

    # histogramas
    histogramaOriginal = calcularHistograma(imagenGris)
    histogramaOtsu = calcularHistograma(imagenOtsu)
    histogramaAdaptativo = calcularHistograma(imagenOtsuAdaptativo)
    rangoNivelesGris = np.arange(256)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    axes[0, 0].imshow(imagenGris, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('Imagen Original en Grises')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(imagenOtsu, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title(f'Otsu Global (Umbral = {umbralGlobal})')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(imagenOtsuAdaptativo, cmap='gray', vmin=0, vmax=255)
    axes[0, 2].set_title(f'Otsu Adaptativo ({anchoVentanaAdaptativo}x{altoVentanaAdaptativo})')
    axes[0, 2].axis('off')

    axes[1, 0].bar(rangoNivelesGris, histogramaOriginal, color='gray')
    axes[1, 0].set_title('Histograma Original')
    axes[1, 0].set_xlabel('Nivel de Gris')
    axes[1, 0].set_ylabel('Frecuencia de Pixeles')
    axes[1, 0].axvline(x=umbralGlobal, color='b', linestyle='--', linewidth=2, label=f'Umbral Otsu = {umbralGlobal}')
    axes[1, 0].legend()
    axes[1, 0].set_xlim([0, 255])

    axes[1, 1].bar(rangoNivelesGris, histogramaOtsu, color='black', width=5.0)
    axes[1, 1].set_title('Histograma Otsu Global')
    axes[1, 1].set_xlabel('Nivel de Gris')
    axes[1, 1].set_xlim([-10, 265])

    axes[1, 2].bar(rangoNivelesGris, histogramaAdaptativo, color='black', width=5.0)
    axes[1, 2].set_title('Histograma Otsu Adaptativo')
    axes[1, 2].set_xlabel('Nivel de Gris')
    axes[1, 2].set_xlim([-10, 265])

    fig.suptitle(f'Resultados de Segmentacion para: {rutaImagen}', fontsize=16)
    plt.show()

if __name__ == "__main__":
    #listaImagenes = ['14.bmp']
    listaImagenes = ['14.bmp', '20.bmp', 'ab1.png']
    anchoVentana = 80
    altoVentana = 80
    
    for nombreImagen in listaImagenes:
        procesarYMostrarImagen(nombreImagen, anchoVentanaAdaptativo=anchoVentana, altoVentanaAdaptativo=altoVentana)
