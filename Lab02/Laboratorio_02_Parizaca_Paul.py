import cv2
import numpy as np
import matplotlib.pyplot as plt

def filtro_media(img, tam_kernel=3):
    img_trans = np.full_like(img, 255, dtype=np.float32)
    desplazamiento = tam_kernel // 2

    for fil in range(desplazamiento, img.shape[0] - desplazamiento):
        for col in range(desplazamiento, img.shape[1] - desplazamiento):
            suma = 0.0
            for i_vec in range(fil - desplazamiento, fil + desplazamiento + 1):
                for j_vec in range(col - desplazamiento, col + desplazamiento + 1):
                    suma += img[i_vec, j_vec]
            img_trans[fil, col] = suma / (tam_kernel * tam_kernel)
    return np.uint8(np.clip(img_trans, 0, 255))

def filtro_mediana(img, tam_kernel=3):
    img_trans = np.full_like(img, 255, dtype=img.dtype) # Usar dtype de img para vecindad
    desplazamiento = tam_kernel // 2
    for fil in range(desplazamiento, img.shape[0] - desplazamiento):
        for col in range(desplazamiento, img.shape[1] - desplazamiento):
            vecindad = []
            for i in range(fil - desplazamiento, fil + desplazamiento + 1):
                for j in range(col - desplazamiento, col + desplazamiento + 1):
                    vecindad.append(img[i, j])
            img_trans[fil, col] = np.median(vecindad)
    return np.uint8(np.clip(img_trans, 0, 255))

def filtro_roberts(img):
    img_trans = np.full_like(img, 255, dtype=np.float32)
    k_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
    k_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
    
    for fil in range(1, img.shape[0]):
        for col in range(1, img.shape[1]):
            gx = 0.0
            gy = 0.0
            for i in range(2):
                for j in range(2):
                    gx += img[fil - 1 + i, col - 1 + j] * k_x[i, j]
                    gy += img[fil - 1 + i, col - 1 + j] * k_y[i, j]
            img_trans[fil, col] = np.sqrt(gx**2 + gy**2)
    return np.uint8(np.clip(img_trans, 0, 255))

def filtro_sobel(img):
    img_trans_x = np.full_like(img, 255, dtype=np.float32)
    img_trans_y = np.full_like(img, 255, dtype=np.float32)
    
    k_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    k_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    for fil in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            gx = 0.0
            gy = 0.0
            for i in range(3):
                for j in range(3):
                    gx += img[fil - 1 + i, col - 1 + j] * k_x[i, j]
                    gy += img[fil - 1 + i, col - 1 + j] * k_y[i, j]
            img_trans_x[fil, col] = gx
            img_trans_y[fil, col] = gy
            
    img_trans = np.sqrt(img_trans_x**2 + img_trans_y**2)
    return np.uint8(np.clip(img_trans, 0, 255))

def filtro_prewitt(img):
    img_trans_x = np.full_like(img, 255, dtype=np.float32)
    img_trans_y = np.full_like(img, 255, dtype=np.float32)
    
    k_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    k_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    
    for fil in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            gx = 0.0
            gy = 0.0
            for i in range(3):
                for j in range(3):
                    gx += img[fil - 1 + i, col - 1 + j] * k_x[i, j]
                    gy += img[fil - 1 + i, col - 1 + j] * k_y[i, j]
            img_trans_x[fil, col] = gx
            img_trans_y[fil, col] = gy
            
    img_trans = np.sqrt(img_trans_x**2 + img_trans_y**2)
    return np.uint8(np.clip(img_trans, 0, 255))

def filtro_laplaciano(img):
    img_trans = np.full_like(img, 255, dtype=np.float32)
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    
    for fil in range(1, img.shape[0] - 1):
        for col in range(1, img.shape[1] - 1):
            suma = 0.0
            for i in range(3):
                for j in range(3):
                    suma += img[fil - 1 + i, col - 1 + j] * kernel[i, j]
            img_trans[fil, col] = suma
    return np.uint8(np.clip(img_trans, 0, 255))


# procesamiento y visualización
nombres_imgs = ['figures.jpg', 'lena.png', 'blancoNegro.png', 'cameraman.jpg', 'figuraV.png', 'figuraV2.jpg', 'gato1.jpg', 'mandrill.jpg', 'ninioB.png', 'patrones1.jpg', 'riceB.png']

for nombre_img in nombres_imgs:
    img = cv2.imread(nombre_img, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: No se pudo cargar la imagen {nombre_img}. Asegúrate de que el archivo exista.")
        continue

    # Filtros de Media y Mediana
    tams_kernel = np.array([3, 5, 7, 9, 11])
    
    figura_mm, ejes_mm = plt.subplots(4, 3, figsize=(10, 18)) # mm: media y mediana
    figura_mm.suptitle(f"Filtros Media y Mediana - {nombre_img}", fontsize=16)

    ejes_mm[0, 0].imshow(img, cmap='gray')
    ejes_mm[0, 0].set_title(f"Original ({nombre_img})")
    ejes_mm[0, 0].axis('off')

    for idx_k, val_tam_k in enumerate(tams_kernel):
        img_media = filtro_media(img, val_tam_k)
        idx_graf = idx_k + 1
        fil_en_bloque = idx_graf // 3
        col_en_bloque = idx_graf % 3
        ejes_mm[fil_en_bloque, col_en_bloque].imshow(img_media, cmap='gray')
        ejes_mm[fil_en_bloque, col_en_bloque].set_title(f"Media {val_tam_k}x{val_tam_k}")
        ejes_mm[fil_en_bloque, col_en_bloque].axis('off')

    ejes_mm[2, 0].imshow(img, cmap='gray')
    ejes_mm[2, 0].set_title(f"Original ({nombre_img})")
    ejes_mm[2, 0].axis('off')

    for idx_k, val_tam_k in enumerate(tams_kernel):
        img_mediana = filtro_mediana(img, val_tam_k)
        idx_graf = idx_k + 1
        fil_en_bloque = idx_graf // 3
        col_en_bloque = idx_graf % 3
        fil_actual = fil_en_bloque + 2 
        ejes_mm[fil_actual, col_en_bloque].imshow(img_mediana, cmap='gray')
        ejes_mm[fil_actual, col_en_bloque].set_title(f"Mediana {val_tam_k}x{val_tam_k}")
        ejes_mm[fil_actual, col_en_bloque].axis('off')
            
    plt.show()

    # Filtros de Detección de Bordes
    img_roberts = filtro_roberts(img)
    img_sobel = filtro_sobel(img)
    img_prewwitt = filtro_prewitt(img)
    img_laplaciano = filtro_laplaciano(img)
    
    figura_bordes, ejes_bordes = plt.subplots(2, 3, figsize=(10, 6))
    figura_bordes.suptitle(f"Filtros Detección de Bordes - {nombre_img}", fontsize=16)

    ejes_bordes[0, 0].imshow(img, cmap='gray')
    ejes_bordes[0, 0].set_title(f"Original ({nombre_img})")
    ejes_bordes[0, 0].axis('off')

    ejes_bordes[0, 1].imshow(img_roberts, cmap='gray')
    ejes_bordes[0, 1].set_title("Filtro Roberts")
    ejes_bordes[0, 1].axis('off')

    ejes_bordes[0, 2].imshow(img_sobel, cmap='gray')
    ejes_bordes[0, 2].set_title("Filtro Sobel")
    ejes_bordes[0, 2].axis('off')

    ejes_bordes[1, 0].imshow(img_prewwitt, cmap='gray')
    ejes_bordes[1, 0].set_title("Filtro Prewitt")
    ejes_bordes[1, 0].axis('off')

    ejes_bordes[1, 1].imshow(img_laplaciano, cmap='gray')
    ejes_bordes[1, 1].set_title("Filtro Laplaciano")
    ejes_bordes[1, 1].axis('off')

    ejes_bordes[1, 2].axis('off')

    plt.show()
