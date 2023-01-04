from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

name_img = glob(os.path.join(os.getcwd(), 'assets/Imagens', '*.jpg'))
font = cv2.FONT_HERSHEY_SIMPLEX
#print(name_img)

for img in name_img:

    areas = list()

    #Pegando imagem de entrada
    img = cv2.imread(img, 1)

    #Separar canais de cores
    B, G, R = cv2.split(img)

    #Diminuir ruidos e realçar bordas
    img_bilateral = cv2.bilateralFilter(G, 1, 90, 90) #Funcionou melhor usando espectro verde
    img_blur = cv2.blur(img_bilateral, (5,5))

    #Binarizar imagem
    img_th = cv2.threshold(img_blur, 190, 255, cv2.THRESH_BINARY)[1]

    #Dilatar imagens
    img_dilate = cv2.dilate(img_th, np.ones((4,4), np.uint8), iterations=1)

    #Aplicando Connected Components para detectar pixels conectados
    numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_dilate, 4, cv2.CV_8U)

    areas.append(stats)
    df_areas = pd.DataFrame(areas[0], columns=['X', 'Y', 'W', 'H', 'AREA'])
    df_areas.drop(df_areas.index[0], inplace=True)
    parafusos = df_areas[df_areas['AREA']>900]
    porcas = df_areas[df_areas['AREA']<899]

    #Mapear rotulos de matiz, 0-179
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    #Converter de HSV PARA BGR
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    #BG preto
    labeled_img[label_hue == 0] = 0
    qtd_elem = numLabels - 1

    flag = True

    if (len(parafusos) != 10):
        print(f'Falta {abs(len(parafusos)-10)} parafusos')
        cv2.putText(
            img, f'Falta {abs(len(parafusos) - 10)} parafusos', (50, 50), font,
            1.5, (255, 255, 255), 2, cv2.LINE_AA)
        flag = False

    if (len(porcas) != 10):
        print(f'Falta {abs(len(porcas)-10)} porcas')
        cv2.putText(
            img, f'Falta {abs(len(porcas)-10)} porcas', (50,50), font,
            1.5, (255,255,255), 2, cv2.LINE_AA)
        flag = False

    if flag == True:
        print('Conjunto Aprovado')
        cv2.putText(
            img, 'Conjunto Aprovado', (200,750), font,
            1.5, (255,255,255), 2, cv2.LINE_AA)

    img_concate = cv2.hconcat([cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cv2.cvtColor(labeled_img, cv2.COLOR_BGR2RGB)])
    img_text = np.zeros((img_concate.shape[0], 50), dtype=np.uint8)
    imagem_total = cv2.hconcat([cv2.cvtColor(img_concate, cv2.COLOR_BGR2RGB), cv2.cvtColor(img_text, cv2.COLOR_BGR2RGB)])
    plt.imshow(imagem_total)
    plt.axis('off')
    plt.title("Imagens")
    plt.show()



