# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 10:47:25 2020

@author: felip
"""

import torch as th
import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

if __name__ == '__main__':

    class Sampler(object):
        """Classe padrão para todos os exemplificadores
        """
    
        def __init__(self, data_source):
            pass
    
        def __iter__(self):
            raise NotImplementedError
    
        def __len__(self):
            raise NotImplementedError
            
    class StratifiedSampler(Sampler):
        """Stratified Sampling
        Provê representação igual para classe selecionada
        """
        def __init__(self, class_vector, controller):            
            self.n_splits = 1
            self.class_vector = class_vector
            self.test_size = test_size
        
        #função para gerar array de exemplos
        def gen_sample_array(self):
            try:
                #tenta importar modelo para pegar imagens aleatorias para separar entre
                #treino e teste
                from sklearn.model_selection import StratifiedShuffleSplit
            except:
                #caso não dê, será exibido esse erro
                print('Need scikit-learn for this functionality')
            
            #utiliza função para separar as imagens
            s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=self.test_size)
            X = th.randn(self.class_vector.size(0),2).numpy()
            y = self.class_vector.numpy()
            s.get_n_splits(X, y)
            
            #define variaveis para treino e teste, atribuindo as imagens selecionadas.
            train_index, test_index= next(s.split(X, y))
            return train_index, test_index
    
        def __iter__(self):
            return iter(self.gen_sample_array())
    
        def __len__(self):
            return len(self.class_vector)
        
    data_dir = "classes"
    
    metadata = pd.read_csv('HAM10000_metadata.csv')
    
    label = [ 'akiec', 'bcc','bkl','df','mel', 'nv',  'vasc']
    classes = [ 'ceratoses actínicas', 'carcinoma basocelular', 'lesoes de ceratose benignas', 
               'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']
    
    def estimar_frequencia(label):
        #DEFINE UM ARRAY DO MESMO TAMANHO QUE O LABEL, APENAS COM ZEROS.
        class_freq = np.zeros_like(label, dtype=np.float)
        #DEFINE O CONTADOR, QUE É UM ARRAY DO MESMO TAMANHO QUE O LABEL, PORÉM VAZIO
        count = np.zeros_like(label)
        for i,l in enumerate(label):
            #DEFINE A FREQUENCIA (QUANTAS IMAGENS) DE CADA CLASSE
            count[i] = metadata[metadata['dx']==str(l)]['dx'].value_counts()[0]
        count = count.astype(np.float)
        #FAZ UMA MEDIA total
        freq_media = np.median(count)
        for i, label in enumerate(label):
            #print(label)
            #DIVIDE A MEDIA TOTAL POR CADA CLASSE, CHEGANDO ASSIM NA FREQUENCIA BALANCEADA.
            class_freq[i] = freq_media / count[i]
        return class_freq
    
    freq = estimar_frequencia(label)
    
    for i in range(len(label)):
        print(label[i],":", freq[i])
    
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2023, 0.1994, 0.2010)
    
    batch_size = 50
    validation_batch_size = 10
    test_batch_size = 10
    
    # Computa a frequencia de cada classe individualmente, e converte para tensors
    class_freq = estimar_frequencia(label)
    class_freq = torch.FloatTensor(class_freq)
    
    transform_train = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=60),
                        transforms.ToTensor(),
                        transforms.Normalize(norm_mean, norm_std),
                        ])
    
    transform_test = transforms.Compose([
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
    
    test_size = 0.2
    val_size = 0.2
    
    #Carrega o dataset
    dataset = torchvision.datasets.ImageFolder(root= data_dir, transform=transform_train)
    #Carrega os labels
    data_label = [s[1] for s in dataset.samples]
    
    #gera o array de exemplos
    ss = StratifiedSampler(torch.FloatTensor(data_label), test_size)
    pre_train_indices, test_indices = ss.gen_sample_array()
    
    #define os indices com os arrays gerados
    train_label = np.delete(data_label, test_indices, None)
    ss = StratifiedSampler(torch.FloatTensor(train_label), test_size)
    train_indices, val_indices = ss.gen_sample_array()
    indices = {'train': pre_train_indices[train_indices],  # Indices of second sampler are used on pre_train_indices
               'val': pre_train_indices[val_indices],  # Indices of second sampler are used on pre_train_indices
               'test': test_indices
               }
    
    # define as variaveis (valores) de cada imagem.
    # Imagens de treino: 6409
    # Imagens de teste 2003
    # Imagens de validação: 1603
    
    train_indices = indices['train']
    val_indices = indices['val']
    test_indices = indices['test']
    print("Imagens de treino:", len(train_indices))
    print("Imagens de teste", len(test_indices))
    print("Imagens de validação:", len(val_indices))
    
    # CARREGAR O DATASET PRA MEMÓRIA
    SubsetRandomSampler = torch.utils.data.sampler.SubsetRandomSampler
    
    train_samples = SubsetRandomSampler(train_indices)
    val_samples = SubsetRandomSampler(val_indices)
    test_samples = SubsetRandomSampler(test_indices)
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=1, sampler= train_samples)
    validation_data_loader = torch.utils.data.DataLoader(dataset, batch_size=validation_batch_size, shuffle=False, sampler=val_samples)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False, sampler=test_samples)
    
    # Função pra mostrar imagem
    fig = plt.figure(figsize=(10, 15))
    def imshow(img):
        img = img / 2 + 0.5     
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # Pegar algumas imagens de treinamento para exibição
    iterador = iter(train_data_loader)
    imagens, labels = iterador.next()
    
    # mostrar tais imagens 
    imshow(torchvision.utils.make_grid(imagens))

