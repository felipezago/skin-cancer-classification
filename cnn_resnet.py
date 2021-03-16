# -*- coding: utf-8 -*-
"""
Created on Tue Nov 6 10:47:25 2020

@author: felip
"""

import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import accuracy_score
import seaborn as sns
from collections import OrderedDict, Sequence
from PIL import Image
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

print(torch.cuda.is_available())
    
def main():
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
    
    num_classes = len(classes)
    
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
    
    #freq = estimar_frequencia(label)
    
    # # for i in range(len(label)):
    # #     print(label[i],":", freq[i])
    
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
    
    # Função pra mostrar imagem
    fig = plt.figure(figsize=(10, 15))
    def imshow(img):
        img = img / 2 + 0.5     
        npimg = img.cpu().numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

    #DEFININDO A REDE NEURAL
    num_classes = len(classes)
    net = torchvision.models.resnet18(pretrained = True)
    
    # We replace last layer of resnet to match our number of classes which is 7
    net.fc = nn.Linear(512, num_classes)
    net = net.to(device)
    
    class_freq = class_freq.to(device)
    criterion = nn.CrossEntropyLoss(weight = class_freq)
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    print(net)
    
    def get_accuracy(predicted, labels):
        batch_len, correct= 0, 0
        batch_len = labels.size(0)
        correct = (predicted == labels).sum().item()
        return batch_len, correct

    def evaluate(model, val_loader):
        losses= 0
        num_samples_total=0
        correct_total=0
        model.eval()
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            _, predicted = torch.max(out, 1)
            loss = criterion(out, labels)
            losses += loss.item() 
            b_len, corr = get_accuracy(predicted, labels)
            num_samples_total +=b_len
            correct_total +=corr
        accuracy = correct_total/num_samples_total
        losses = losses/len(val_loader)
        return losses, accuracy
    
    # num_epochs = 1
    # accuracy = []
    # val_accuracy = []
    # losses = []
    # val_losses = []
    
    # for epoch in range(num_epochs):
    #     running_loss = 0.0
    #     correct_total= 0.0
    #     num_samples_total=0.0
    #     for i, data in enumerate(train_data_loader):
    #         # get the inputs
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         # set the parameter gradients to zero
    #         optimizer.zero_grad()
    
    #         # forward + backward + optimize
    #         outputs = net(inputs)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
            
    #         #compute accuracy
    #         _, predicted = torch.max(outputs, 1)
    #         b_len, corr = get_accuracy(predicted, labels)
    #         num_samples_total +=b_len
    #         correct_total +=corr
    #         running_loss += loss.item()
    
        
    #     running_loss /= len(train_data_loader)
    #     train_accuracy = correct_total/num_samples_total
    #     val_loss, val_acc = evaluate(net, validation_data_loader)
        
    #     print('Epoch: %d' %(epoch+1))
    #     print('Loss: %.3f  Accuracy:%.3f' %(running_loss, train_accuracy))
    #     print('Validation Loss: %.3f  Val Accuracy: %.3f' %(val_loss, val_acc))
    
    #     losses.append(running_loss)
    #     val_losses.append(val_loss)
    #     accuracy.append(train_accuracy)
    #     val_accuracy.append(val_acc)
    # print('Finished Training')
    
    PATH = 'model.pth'
    #####################3torch.save(net.state_dict(), PATH)
    
    # epoch = range(1, num_epochs+1)
    # fig = plt.figure(figsize=(10, 15))
    # plt.subplot(2,1,2)
    # plt.plot(epoch, losses, label='Training loss')
    # plt.plot(epoch, val_losses, label='Validation loss')
    # plt.title('Training and Validation Loss')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.figure()
    # plt.show()
    
    # fig = plt.figure(figsize=(10, 15))
    # plt.subplot(2,1,2)
    # plt.plot(epoch, accuracy, label='Training accuracy')
    # plt.plot(epoch, val_accuracy, label='Validation accuracy')
    # plt.title('Training and Validation Accuracy')
    # plt.xlabel('Epochs')
    # plt.legend()
    # plt.figure()
    # plt.show()
    
    # fig = plt.figure(figsize=(10, 15))
    
    dataiter = iter(test_data_loader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    trans = transforms.Compose([
                        transforms.Resize((7,7)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=60),
                        transforms.ToTensor(),
                        transforms.Normalize(norm_mean, norm_std),
                        ])
            
   # print images
    imshow(torchvision.utils.make_grid(images))
    print('Classe Real: ', ' '.join('%7s' % classes[labels[j]] for j in range(4)))
    
    net = torchvision.models.resnet18(pretrained = True)

    net.fc = nn.Linear(512, num_classes)
    net = net.to(device)
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    image = Image.open(Path('classes/df/ISIC_0024318.jpg')) 
    transformed= trans(image).unsqueeze_(0)
    outputs = net(images)

    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    # print(classes[predicted])


    print('Predito: ', ' '.join('%7s' % classes[predicted[j]]
                                  for j in range(4)))  

    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(3):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    
    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    
    # correct = 0
    # total = 0
    # net.eval()
    # with torch.no_grad():
    #     for data in test_data_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    
    # print('Accuracy of the network on the test images: %d %%' % (
    #     100 * correct / total))
    
    # class_correct = list(0. for i in range(len(classes)))
    # class_total = list(1e-7 for i in range(len(classes)))
    # with torch.no_grad():
    #     for data in test_data_loader:
    #         images, labels = data
    #         images, labels = images.to(device), labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(3):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    
    # for i in range(len(classes)):
    #     print('Accuracy of %5s : %2d %%' % (
    #         classes[i], 100 * class_correct[i] / class_total[i]))

    # image = Image.open(Path('classes/df/ISIC_0024318.jpg'))        
        
    confusion_matrix = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1
    
    print(confusion_matrix)
    cm = confusion_matrix.numpy()    
    
    fig,ax= plt.subplots(figsize=(7,7))
    #sns.heatmap(cm / (cm.astype(np.float).sum(axis=1) + 1e-9), annot=False, ax=ax)
    
    # labels, title and ticks
    ax.set_xlabel('Predicted', size=25);
    ax.set_ylabel('True', size=25); 
    ax.set_title('Confusion Matrix', size=25); 
    ax.xaxis.set_ticklabels(['akiec','bcc','bkl','df', 'mel', 'nv','vasc'], size=15); \
    ax.yaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','nv','vasc'], size=15);


if __name__ == '__main__':
    main()