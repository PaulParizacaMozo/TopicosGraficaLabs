import cv2
import numpy as np
import matplotlib.pyplot as plt

def erosion(img, kernel, nucleo_pos):
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    kx, ky = nucleo_pos
    
    erosion_result = np.full_like(img, 255)
    for i in range(kx, img_height - (kernel_height - kx)):
        for j in range(ky, img_width - (kernel_width - ky)):
            region = img[i - kx:i - kx + kernel_height, j - ky:j - ky + kernel_width]
            if np.all(region[kernel == 1] == 255):
                erosion_result[i, j] = 255
            else:
                erosion_result[i, j] = 0
    return erosion_result

def dilatacion(img, kernel, nucleo_pos):
    img_height, img_width = img.shape
    kernel_height, kernel_width = kernel.shape
    kx, ky = nucleo_pos
    
    dilatacion_result = np.full_like(img, 255)
    
    for i in range(kx, img_height - (kernel_height - kx)):
        for j in range(ky, img_width - (kernel_width - ky)):
            region = img[i - kx:i - kx + kernel_height, j - ky:j - ky + kernel_width]
            if np.any(region[kernel == 1] == 255):
                dilatacion_result[i, j] = 255
            else:
                dilatacion_result[i, j] = 0
    return dilatacion_result

def binarizar(img, umbral=128):
    img_height, img_width = img.shape
    img_binarizada = np.zeros_like(img)
    
    for i in range(img_height):
        for j in range(img_width):
            if img[i, j] >= umbral:
                img_binarizada[i, j] = 255
            else:
                img_binarizada[i, j] = 0
    return img_binarizada

# kernels y nucleo(centro para todos)
kernels = [
    (np.array([[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]], dtype=np.uint8), (1, 1), "Cruz 3x3"),
    (np.array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]], dtype=np.uint8), (2, 2), "Diamante 5x5"),
    (np.array([[0, 0, 0, 1, 0, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1, 1],
               [0, 1, 1, 1, 1, 1, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8), (3, 3), "Diamante 7x7"),
    (np.array([[1],
               [1],
               [1],
               [1],
               [1]], dtype=np.uint8), (2, 0), "Barra Vertical 5x1"),
    (np.array([[0, 1, 0]], dtype=np.uint8), (0, 1), "Barra Horizontal 1x3")
]

# imagenes de prueba
image_files = ['figures.jpg', 'blancoNegro.png', 'figuraV2.jpg', 'figuraV.png', 'gato1.jpg']
#image_files = ['figuraV.png']

# plots de resultados
for image_file in image_files:
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"No se pudo cargar la imagen '{image_file}'. Asegúrate de que la imagen esté en el directorio correcto.")
        continue
    
    # binarizacion de la imagen
    img_binarizada = binarizar(img, umbral=128)
    
    # erosion y dilatacion  usando cada kernel
    erosion_results = []
    dilatacion_results = []
    for kernel, nucleo_pos, _ in kernels:
        erosion_results.append(erosion(img_binarizada, kernel, nucleo_pos))
        dilatacion_results.append(dilatacion(img_binarizada, kernel, nucleo_pos))
    
    # mostrar resultados
    for operation, results, title_prefix in [
        ("Erosión", erosion_results, f"Erosión ({image_file})"),
        ("Dilatación", dilatacion_results, f"Dilatación ({image_file})")
    ]:
        plt.figure(figsize=(10, 6))
        
        # Imagen binarizada
        plt.subplot(2, 3, 1)
        plt.imshow(img_binarizada, cmap='gray')
        plt.title('Imagen Binarizada')
        plt.axis('off')
        
        # Resultados de la operación con cada kernel
        for i, (result, (_, _, kernel_name)) in enumerate(zip(results, kernels), start=2):
            plt.subplot(2, 3, i)
            plt.imshow(result, cmap='gray')
            plt.title(f'{title_prefix} - {kernel_name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
