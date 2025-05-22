import cv2
import numpy as np
import matplotlib.pyplot as plt

def filtro_sobel(img_entrada):
    img = img_entrada.astype(np.float32)

    img_trans_x = np.zeros_like(img, dtype=np.float32)
    img_trans_y = np.zeros_like(img, dtype=np.float32)

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
    img_resultado = np.uint8(np.clip(img_trans, 0, 255))
    
    return img_resultado

def filtro_prewitt(img_entrada):
    img = img_entrada.astype(np.float32)

    img_trans_x = np.zeros_like(img, dtype=np.float32)
    img_trans_y = np.zeros_like(img, dtype=np.float32)

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
    img_resultado = np.uint8(np.clip(img_trans, 0, 255))
    
    return img_resultado

def filtro_roberts(img_entrada):
    img = img_entrada.astype(np.float32)

    img_trans_x = np.zeros_like(img, dtype=np.float32)
    img_trans_y = np.zeros_like(img, dtype=np.float32)

    # Nuevas matrices para el eje X y eje Y
    k_x = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
    k_y = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)

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
    img_resultado = np.uint8(np.clip(img_trans, 0, 255))
    
    return img_resultado

# aplicar el filtro en cada canal RGB
def aplicar_filtro_rgb(img, filtro):
    b, g, r = cv2.split(img)

    # filtro a cada canal
    b_procesado = filtro(b)
    g_procesado = filtro(g)
    r_procesado = filtro(r)

    canales_procesados = [b_procesado, g_procesado, r_procesado]
    
    alto, ancho = b_procesado.shape

    # matriz 3D
    img_procesada = np.zeros((alto, ancho, 3), dtype=np.uint8)

    img_procesada[:, :, 0] = b_procesado
    img_procesada[:, :, 1] = g_procesado
    img_procesada[:, :, 2] = r_procesado
    
    return canales_procesados, img_procesada

# tests
nombres_imgs = ['lena.png', 'mandrill.jpg','patrones1.jpg']

filtros = [filtro_sobel, filtro_prewitt, filtro_roberts]

# Plots
for filtro in filtros:
    for nombre_img in nombres_imgs:
        img_original_color = cv2.imread(nombre_img)
        
        if img_original_color is None:
            print(f"No se pudo cargar la imagen {nombre_img}.")
            continue

        canales, img_mezclado = aplicar_filtro_rgb(img_original_color, filtro)

        # 2 filas, 3 columnas
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f"Procesamiento con {filtro.__name__} para: {nombre_img}", fontsize=16)

        # Mostrar la imagen original
        axes[0, 0].imshow(cv2.cvtColor(img_original_color, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Original")
        axes[0, 0].axis('off')

        # Mcanal rojo
        axes[0, 1].imshow(canales[2], cmap='Reds')
        axes[0, 1].set_title(f"Canal Rojo")
        axes[0, 1].axis('off')

        # canal verde
        axes[0, 2].imshow(canales[1], cmap='Greens')
        axes[0, 2].set_title(f"Canal Verde")
        axes[0, 2].axis('off')

        # canal azul
        axes[1, 0].imshow(canales[0], cmap='Blues')
        axes[1, 0].set_title(f"Canal Azul")
        axes[1, 0].axis('off')

        # combinación de los canales
        axes[1, 1].imshow(cv2.cvtColor(img_mezclado, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title(f"Combinado")
        axes[1, 1].axis('off')

        # celda vacía
        axes[1, 2].axis('off')

        plt.show()
