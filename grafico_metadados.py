# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:42:42 2020

@author: felip
"""

import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import torch
    
metadata = pd.read_csv('HAM10000_metadata.csv')
print(metadata.shape)

print(torch.cuda.FloatTensor())

# armazena as sete classes do dataset na variável "le"

le = LabelEncoder()
le.fit(metadata['dx'])
LabelEncoder()
# printa todas as classes
print("Classes:", list(le.classes_))

metadata['label'] = le.transform(metadata["dx"]) 
metadata.sample(10)

#define o tamanho da figura
fig = plt.figure(figsize=(40,25))

#faz um gráfico referente ao tipo de cada cancer

#posição do gráfico
ax1 = fig.add_subplot(221)
#valida as informações do campo "dx", referente aos tipos de cancer, no arquivo de metadados
# também valida o tipo do gráfico, no caso, de barra
metadata['dx'].value_counts().plot(kind='bar', ax=ax1)
#informa o titulo das informações no eixo Y
ax1.set_ylabel('Contagem', size=50)
#informa o titulo do gráfico
ax1.set_title('Tipo de cancer', size = 50)

ax2 = fig.add_subplot(222)
metadata['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Contagem', size=50)
ax2.set_title('Sexo', size=50);

ax3 = fig.add_subplot(223)
metadata['localization'].value_counts().plot(kind='bar')
ax3.set_ylabel('Contagem', size=50)
ax3.set_title('Local do corpo', size=50)

ax4 = fig.add_subplot(224)
sample_age = metadata[pd.notnull(metadata['age'])]
sns.distplot(sample_age['age'], fit=stats.norm, color='red');
ax4.set_title('Idade', size = 50)
ax4.set_xlabel('Ano', size=50)

plt.tight_layout()
plt.show()