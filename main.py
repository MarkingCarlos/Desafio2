from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

#Aplica limiar de Otsu e Watershad
def OtsuWater(img):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # limiarização de Otsu
    _, otsu_thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    
    # Remove o ruído
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Determina o fundo certo
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    
    # Determina a área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Encontrar regiões desconhecidas
    sure_fg = np.uint8(sure_fg)
    RegiaoDesconhecida = cv2.subtract(sure_bg, sure_fg)
    
    # Marcação de marcador
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[RegiaoDesconhecida == 255] = 0
    
    # Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    
    return img, otsu_thresh, markers, gray

def OtsuWaterBox(img):
    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # limiarização de Otsu
    _, otsu_thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY)
    
    # Remove o ruído
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Determina o fundo certo
    sure_bg = cv2.dilate(opening, kernel, iterations=2)
    
    # Determina a área de primeiro plano
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    
    # Encontrar regiões desconhecidas
    sure_fg = np.uint8(sure_fg)
    RegiaoDesconhecida = cv2.subtract(sure_bg, sure_fg)
    
    # Marcação de marcador
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[RegiaoDesconhecida == 255] = 0
    
    # Watershed
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]
    
    return markers, gray

def GLCM(gray):
    # Normalizar a imagem para os níveis de cinza entre 254 e 255
    gray_normalized = cv2.normalize(gray, None, 254, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calcular a matriz de co-ocorrência para d=1 e ângulos 0, 90, 180, 270
    distances = [1]
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    glcm = graycomatrix(gray_normalized, distances, angles, 256, symmetric=True, normed=True)
    
    return glcm

def informacao(glcm):
    descriptors = {}
    descriptors['Segundo Momento Angular'] = graycoprops(glcm, 'energy').mean()
    descriptors['Correlação'] = graycoprops(glcm, 'correlation').mean()
    descriptors['Contraste'] = graycoprops(glcm, 'contrast').mean()
    descriptors['Homogeneidade'] = graycoprops(glcm, 'homogeneity').mean()
    descriptors['Entropia'] = shannon_entropy(glcm)
    return descriptors

def DimensaoFraquetal(Z):

    assert(len(Z.shape) == 2)

    def box_count(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)

        
        return len(np.where((S > 0) & (S < k*k))[0])

    # 
    Z = (Z < 255)

    # dimensão minima
    p = min(Z.shape)

    # Maior potência de 2 menor ou igual a p
    n = 2**np.floor(np.log2(p))

    # expoente
    n = int(np.log2(n))

    sizes = 2**np.arange(n, 1, -1)

    # Contagem de caixas com tamanho decrescente
    counts = []
    for size in sizes:
        counts.append(box_count(Z, size))

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def compute_feature_vector(gray, mask):
    MascaraCinza = gray * mask
    glcm = GLCM(MascaraCinza)
    descriptors = informacao(glcm)
    DF = DimensaoFraquetal(MascaraCinza)
    feature_vector = [
        descriptors['Segundo Momento Angular'],
        descriptors['Entropia'],
        descriptors['Correlação'],
        descriptors['Contraste'],
        descriptors['Homogeneidade'],
        DF
    ]
    return feature_vector

def box_count(img, k):
    S = np.add.reduceat(
        np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
        np.arange(0, img.shape[1], k), axis=1)

    return len(np.where((S > 0) & (S < k*k))[0])

def plot_feature_spaces(Normal, Media, Alta):
    # Nomes das características
    feature_names = ["Segundo Momento Angular", "Entropia", "Correlação", "Contraste", "Homogeneidade", "Dimensão Fractal (DF)"]
    
    # Número de características
    num_features = len(feature_names)
    
    # Criando subplots
    fig, axs = plt.subplots(num_features, num_features, figsize=(20, 20))
    
    # Plotando os espaços de características
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                axs[i, j].scatter(Normal[:, j], Normal[:, i], color='b', label='Normal', alpha=0.5)
                axs[i, j].scatter(Media[:, j], Media[:, i], color='g', label='Média', alpha=0.5)
                axs[i, j].scatter(Alta[:, j], Alta[:, i], color='r', label='Alta', alpha=0.5)
                axs[i, j].set_xlabel(feature_names[j])
                axs[i, j].set_ylabel(feature_names[i])
            else:
                axs[i, j].hist(Normal[:, i], color='b', alpha=0.5, label='Normal')
                axs[i, j].hist(Media[:, i], color='g', alpha=0.5, label='Média')
                axs[i, j].hist(Alta[:, i], color='r', alpha=0.5, label='Alta')
                axs[i, j].set_xlabel(feature_names[i])
                axs[i, j].set_ylabel('Frequência')
    
    # Adicionando a legenda
    handles, labels = axs[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.show()

def main(img):

    segmented_img,otsu_thresh, markers, gray = OtsuWater(img)
    
    # Cálculo de características para objeto e fundo
    object_mask = (markers == 1).astype(np.uint8)
    
    object_feature_vector = compute_feature_vector(gray, object_mask)
    
    # Mostrando segmentações
    fig, axs = plt.subplots(1, 2, figsize=(20, 5))
    axs[0].imshow(otsu_thresh, cmap='gray')
    axs[1].set_title('Limiarização de Otsu')
    axs[1].axis('off')

    axs[1].imshow(cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB))
    axs[1].set_title('Segmentação Watershed')
    axs[1].axis('off')

    plt.show() 
   
    # Exibindo os resultados

    # Imprimir os vetores de características
    print("Vetor de Características do Objeto:")
    print("Segundo Momento Angular:", object_feature_vector[0])
    print("Entropia:", object_feature_vector[1])
    print("Correlação:", object_feature_vector[2])
    print("Contraste:", object_feature_vector[3])
    print("Homogeneidade:", object_feature_vector[4])
    print("Dimensão Fractal (DF):", object_feature_vector[5])

    
def mainBox(normal,media, alta):

    #Grafico imagem normal
    markers, gray = OtsuWaterBox(normal)
    object_mask = (markers == 1).astype(np.uint8)
    object_feature_vector = compute_feature_vector(gray, object_mask)
    Normalobject_features = np.array(object_feature_vector).reshape(1, -1)

    #Grafico imagem 44h
    markers, gray = OtsuWaterBox(media)
    object_mask = (markers == 1).astype(np.uint8)
    object_feature_vector = compute_feature_vector(gray, object_mask)
    Mediaobject_features = np.array(object_feature_vector).reshape(1, -1)

    #Grafico imagem 96h
    markers, gray = OtsuWaterBox(alta)
    object_mask = (markers == 1).astype(np.uint8)
    object_feature_vector = compute_feature_vector(gray, object_mask)
    altaobject_features = np.array(object_feature_vector).reshape(1, -1)

    # Exibindo os resultados
    plot_feature_spaces(Normalobject_features, Mediaobject_features, altaobject_features)

   

if __name__ == "__main__":
    image_path = "Normal.jpg"
    Normalimg = cv2.imread(image_path)
    print("Imagem Normal:")
    main(Normalimg)
    image_path = "44h.jpg"
    Mediaimg = cv2.imread(image_path)
    print("Imagem 44h:")
    main(Mediaimg)
    image_path = "96h.jpg"
    Altaimg = cv2.imread(image_path)
    print("Imagem 96h:")
    main(Altaimg)
    mainBox(Normalimg,Mediaimg,Altaimg)
   