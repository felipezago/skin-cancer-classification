# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 12:51:14 2020

@author: felip
"""
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import camskin.cnn_resnet as c
import numpy as np

classes = [ 'ceratoses actínicas', 'carcinoma basocelular', 'lesoes de ceratose benignas', 
               'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']
    
num_classes = len(classes)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if __name__ == '__main__':
    
    data_dir = "arquivos/classes"

    
    label = [ 'akiec', 'bcc','bkl','df','mel', 'nv',  'vasc']
    classes = [ 'ceratoses actínicas', 'carcinoma basocelular', 'lesoes de ceratose benignas', 
               'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']
    
    num_classes = len(classes)
    
    norm_mean = (0.4914, 0.4822, 0.4465)
    norm_std = (0.2023, 0.1994, 0.2010)
    
    batch_size = 50
    validation_batch_size = 10
    test_batch_size = 10
    
    # Computa a frequencia de cada classe individualmente, e converte para tensors
    
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
    ss = c.StratifiedSampler(torch.FloatTensor(data_label), test_size)
    pre_train_indices, test_indices = ss.gen_sample_array()
    
    #define os indices com os arrays gerados
    train_label = np.delete(data_label, test_indices, None)
    ss = c.StratifiedSampler(torch.FloatTensor(train_label), test_size)
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
    # print("Imagens de treino:", len(train_indices))
    # print("Imagens de teste", len(test_indices))
    # print("Imagens de validação:", len(val_indices))
    
    # CARREGAR O DATASET PRA MEMÓRIA
    SubsetRandomSampler = torch.utils.data.sampler.SubsetRandomSampler
    
    train_samples = SubsetRandomSampler(train_indices)
    val_samples = SubsetRandomSampler(val_indices)
    test_samples = SubsetRandomSampler(test_indices)
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=1, sampler= train_samples)
    validation_data_loader = torch.utils.data.DataLoader(dataset, batch_size=validation_batch_size, shuffle=False, sampler=val_samples)
    test_data_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False, sampler=test_samples)
    
    net = torchvision.models.resnet18(pretrained = True)
    net.fc = nn.Linear(512, num_classes)
    net = net.to(device)

    PATH = 'arquivos/model.pth'
    model = net.load_state_dict(torch.load(PATH, map_location=device))

    
    

    