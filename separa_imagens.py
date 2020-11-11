import os
import pandas as pd

# le o arquivo CSV
metadados = pd.read_csv('HAM10000_metadata.csv')

# Separa um array com todas as classes
classes = set([x[1]['dx'] for x in metadados.iterrows()])

print(classes)

# cria os diretórios com o nome das classes
for classe in classes:
    os.mkdir('classes/{}'.format(classe))

# retira as imagens do diretório que elas estão, e coloca em sua respectiva pasta, com base na sua classe
for idx, linha in metadados.iterrows():    
    os.rename('ham10000/imagens/{}.jpg'.format(linha['image_id']), 'classes/{}/{}.jpg'.format(linha['dx'], linha['image_id']))