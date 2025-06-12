import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def funHistograma(img):
    nivelesIntensidad = 256
    histograma = np.zeros(nivelesIntensidad, dtype=int)
    filas, columnas = img.shape

    for i in range(filas):
        for j in range(columnas):
            valorPixel = img[i, j]
            histograma[valorPixel] += 1
 
    return histograma

def convolucion2d(img, kernel):
    kernelAlt, kernelAnch = kernel.shape
    padAlt, padAnch = kernelAlt // 2, kernelAnch // 2
    imgAlt, imgAnch = img.shape
 
    # padding
    imgConPad = np.zeros((imgAlt + 2*padAlt, imgAnch + 2*padAnch))
    imgConPad[padAlt:imgAlt+padAlt, padAnch:imgAnch+padAnch] = img
 
    salida = np.zeros_like(img, dtype=np.float64)

    # convolucion
    for i in range(imgAlt):
        for j in range(imgAnch):
            region = imgConPad[i : i + kernelAlt, j : j + kernelAnch]
            salida[i, j] = np.sum(region * kernel)

    return salida

def filtroGaussiano(img, tamKernel=5, sigma=1.4):
    kernel = np.zeros((tamKernel, tamKernel))
    centro = tamKernel // 2
 
    for x in range(tamKernel):
        for y in range(tamKernel):
            dx = x - centro
            dy = y - centro
            kernel[x, y] = (1 / (2 * math.pi * sigma**2)) * math.exp(- (dx**2 + dy**2) / (2 * sigma**2))
 
    # normalizar kernel
    kernel /= np.sum(kernel)
 
    # convolucion
    imgSuavizada = convolucion2d(img, kernel)
    return imgSuavizada

def filtroSobel(img):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
 
    # calcular gradientes
    gradX = convolucion2d(img, kx)
    gradY = convolucion2d(img, ky)
 
    return gradX, gradY

def detectorCanny(img, ratioUmbralBajo=0.05, ratioUmbralAlto=0.12):
    # reduccion de ruido
    imgSuavizada = filtroGaussiano(img.astype(np.float64))
 
    # filtro Sobel
    gradX, gradY = filtroSobel(imgSuavizada)
    magnitud = np.sqrt(gradX**2 + gradY**2)
    filas, columnas = img.shape
 
    # supresion de no maximos
    suprimida = np.zeros_like(magnitud)

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            q, r = 255.0, 255.0
            direccionPixel = math.atan2(gradY[i, j], gradX[i, j])
            anguloPixel = direccionPixel * 180. / math.pi
            if anguloPixel < 0: anguloPixel += 180

            if (0 <= anguloPixel < 45):
                d = math.tan(direccionPixel)
                q = magnitud[i, j+1] * (1-d) + magnitud[i+1, j+1] * d
                r = magnitud[i, j-1] * (1-d) + magnitud[i-1, j-1] * d
            elif (45 <= anguloPixel < 90):
                d = 1/math.tan(direccionPixel)
                q = magnitud[i+1, j] * (1-d) + magnitud[i+1, j+1] * d
                r = magnitud[i-1, j] * (1-d) + magnitud[i-1, j-1] * d
            elif (90 <= anguloPixel < 135):
                d = -1/math.tan(direccionPixel)
                q = magnitud[i+1, j] * (1-d) + magnitud[i+1, j-1] * d
                r = magnitud[i-1, j] * (1-d) + magnitud[i-1, j+1] * d
            else:
                d = -math.tan(direccionPixel)
                q = magnitud[i, j-1] * (1-d) + magnitud[i+1, j-1] * d
                r = magnitud[i, j+1] * (1-d) + magnitud[i-1, j+1] * d

            if (magnitud[i, j] >= q) and (magnitud[i, j] >= r):
                suprimida[i, j] = magnitud[i, j]
 
    # doble umbral e histeresis
    umbralAlto = np.max(suprimida) * ratioUmbralAlto
    umbralBajo = umbralAlto * ratioUmbralBajo

    valFuerte = 255
    valDebil = 75
 
    coordsFuertes = []
    coordsDebiles = []
    for i in range(filas):
        for j in range(columnas):
            if suprimida[i, j] >= umbralAlto:
                coordsFuertes.append((i, j))
            elif suprimida[i, j] >= umbralBajo:
                coordsDebiles.append((i, j))
 
    resultado = np.zeros((filas, columnas), dtype=np.uint8)
    for i, j in coordsFuertes:
        resultado[i, j] = valFuerte
    for i, j in coordsDebiles:
        resultado[i, j] = valDebil

    for i in range(1, filas - 1):
        for j in range(1, columnas - 1):
            if resultado[i, j] == valDebil:
                if valFuerte in resultado[i-1:i+2, j-1:j+2]:
                    resultado[i, j] = valFuerte
                else:
                    resultado[i, j] = 0
    # imgs
    return imgSuavizada, magnitud, suprimida, resultado

if __name__ == "__main__":
    listaImagenes = ['Lab2.jpg', 'Lab03.jpg', 'Lab04.jpg', 'lab05.jpg']
    for rutaImagen in listaImagenes:
        print(f"imagen: {rutaImagen}")
        img = cv2.imread(rutaImagen, cv2.IMREAD_GRAYSCALE)
        # imgs resultantes
        imgSuavizada, magnitud, imgSuprimida, bordesFinales = detectorCanny(img,0.1,0.2)
 
        # plot 1
        plt.figure(figsize=(18, 10))
        plt.suptitle(f"Pasos del Algoritmo Canny para: {rutaImagen}", fontsize=16)

        plt.subplot(2, 3, 1)
        plt.imshow(img, cmap='gray')
        plt.title('1. Imagen Original')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(imgSuavizada, cmap='gray')
        plt.title('2. Imagen Suavizada(Gaussiano)')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(magnitud, cmap='gray')
        plt.title('3. Magnitud del Gradiente(Sobel)')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(imgSuprimida, cmap='gray')
        plt.title('4. Supresion No Maximos')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(bordesFinales, cmap='gray')
        plt.title('5. Resultado Final Canny')
        plt.axis('off')
        
        plt.subplot(2, 3, 6).axis('off')

        # plot2
        histOriginal = funHistograma(img)
        histBordes = funHistograma(bordesFinales)
        
        plt.figure(figsize=(12, 10))
        plt.suptitle(f"Histogramas para: {rutaImagen}", fontsize=16)
        
        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.title('Imagen Original')
        plt.axis('off')

        plt.subplot(2, 2, 2)
        plt.imshow(bordesFinales, cmap='gray')
        plt.title('Resultado Final Canny')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.bar(range(256), histOriginal, color='gray')
        plt.title('Histograma de la Imagen Original')
        plt.xlabel('Nivel de Intensidad')
        plt.ylabel('Frecuencia')
        plt.xlim([-5, 260])

        plt.subplot(2, 2, 4)
        plt.bar(range(256), histBordes, color='black', width=5.0)
        plt.title('Histograma del Resultado Final')
        plt.xlabel('Nivel de Intensidad (0 y 255)')
        plt.ylabel('Frecuencia de Pixeles')
        plt.xlim([-10, 265]) 
        
        plt.show()



