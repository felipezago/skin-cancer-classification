# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:45:36 2020

@author: felip
"""

import pandas as pd
import imageio
import matplotlib.pyplot as plt


data_dir = "classes"
    
metadata = pd.read_csv('HAM10000_metadata.csv')  

label = [ 'akiec', 'bcc','bkl','df','mel', 'nv',  'vasc']
label_imagens = []
classes = [ 'ceratoses actínicas', 'carcinoma basocelular', 'lesoes de ceratose benignas', 
           'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']

#Define o tamanho da figura
fig = plt.figure(figsize=(55, 55))
k = range(7)

#Separa 5 imagens de cada classe e armazena em ordem no array label_imagens
for i in label:
    imgs = metadata[metadata['dx'] == i]['image_id'][:5]
    label_imagens.extend(imgs)
    
#Seleciona cada descrição no array label_imagens
for posicao,ID in enumerate(label_imagens):
    #define a classe de cada imagem na variavel lbl
    labl = metadata[metadata['image_id'] == ID]['dx']
    
    # busca as imagens no diretório definido, com base no ID
    imagem = data_dir + "/" + labl.values[0] + f'/{ID}.jpg'
    imagem = imageio.imread(imagem)
    
    plt.subplot(7,5,posicao+1)
    plt.imshow(imagem)
    plt.axis('off')

    #Define o titulo de cada linha, com base no array classes.
    if posicao%5 == 0:
        title = int(posicao/5)
        plt.title(classes[title], loc='center', size=50, weight="bold")
       
#gera a imagem
plt.tight_layout()
plt.show()

