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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    freq = estimar_frequencia(label)
    
    # for i in range(len(label)):
    #     print(label[i],":", freq[i])
    
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
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    # Pegar algumas imagens de treinamento para exibição
    iterador = iter(train_data_loader)
    imagens, labels = iterador.next()
    
    # mostrar tais imagens 
    imshow(torchvision.utils.make_grid(imagens))
    
    for j in range(len(labels)):
        print(labels[j].to(int))

    #DEFININDO A REDE NEURAL
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, (5,5), padding=2)
            self.conv2 = nn.Conv2d(6, 16, (5,5))
            self.fc1   = nn.Linear(16*54*54, 120)
            self.fc2   = nn.Linear(120, 84)
            self.fc3   = nn.Linear(84, num_classes)
        def forward(self, x):
            x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
            x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
            x = x.view(-1, self.num_flat_features(x))
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
        def num_flat_features(self, x):
            size = x.size()[1:]
            num_features = 1
            for s in size:
                num_features *= s
            return num_features
    
    net = LeNet()
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

    # COMEÇO DO TREINAMENTO DA REDE NEURAL
    num_epochs = 1
    accuracy = []
    val_accuracy = []
    losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_total= 0.0
        num_samples_total=0.0
        for i, data in enumerate(train_data_loader):
            # get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # set the parameter gradients to zero
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #compute accuracy
            _, predicted = torch.max(outputs, 1)
            b_len, corr = get_accuracy(predicted, labels)
            num_samples_total +=b_len
            correct_total +=corr
            running_loss += loss.item()
    
        
        running_loss /= len(train_data_loader)
        train_accuracy = correct_total/num_samples_total
        val_loss, val_acc = evaluate(net, validation_data_loader)
        
        print('Epoch: %d' %(epoch+1))
        print('Loss: %.3f  Accuracy:%.3f' %(running_loss, train_accuracy))
        print('Validation Loss: %.3f  Val Accuracy: %.3f' %(val_loss, val_acc))
    
        losses.append(running_loss)
        val_losses.append(val_loss)
        accuracy.append(train_accuracy)
        val_accuracy.append(val_acc)
    print('Finished Training')


    epoch = range(1, num_epochs+1)
    fig = plt.figure(figsize=(10, 15))
    plt.subplot(2,1,2)
    plt.plot(epoch, losses, label='Training loss')
    plt.plot(epoch, val_losses, label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.figure()
    plt.show()
    
    fig = plt.figure(figsize=(10, 15))
    plt.subplot(2,1,2)
    plt.plot(epoch, accuracy, label='Training accuracy')
    plt.plot(epoch, val_accuracy, label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.figure()
    plt.show()
    
    fig = plt.figure(figsize=(10, 15))
    dataiter = iter(test_data_loader)
    images, labels = dataiter.next()
    
    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s,  ' % classes[labels[j]] for j in range(len(labels))))
    
    # testar a precisão
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
    
    print('Accuracy of the network on the test images: %d %%' % (
        100 * correct / total))
    
    # testar a precisão de cada classe individualmente
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(1e-7 for i in range(len(classes)))
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
        
        
    #matriz de confusão
    confusion_matrix = torch.zeros(len(classes), len(classes))
    with torch.no_grad():
        for data in test_data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), predicted.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

    cm = confusion_matrix.numpy()
    fig,ax= plt.subplots(figsize=(7,7))
    sns.heatmap(cm / (cm.astype(np.float).sum(axis=1) + 1e-9), annot=False, ax=ax)
    
    # labels, title and ticks
    ax.set_xlabel('Predicted', size=25);
    ax.set_ylabel('True', size=25); 
    ax.set_title('Confusion Matrix', size=25); 
    ax.xaxis.set_ticklabels(['akiec','bcc','bkl','df', 'mel', 'nv','vasc'], size=15); \
    ax.yaxis.set_ticklabels(['akiec','bcc','bkl','df','mel','nv','vasc'], size=15);
    
    # gradcam
    class _BaseWrapper(object):
        """
        Please modify forward() and backward() according to your task.
        """
    
        def __init__(self, model):
            super(_BaseWrapper, self).__init__()
            self.device = next(model.parameters()).device
            self.model = model
            self.handlers = []  # a set of hook function handlers
    
        def _encode_one_hot(self, ids):
            one_hot = torch.zeros_like(self.logits).to(self.device)
            one_hot.scatter_(1, ids, 1.0)
            return one_hot
    
        def forward(self, image):
            """
            Simple classification
            """
            self.model.zero_grad()
            self.logits = self.model(image)
            self.probs = F.softmax(self.logits, dim=1)
            return self.probs.sort(dim=1, descending=True)
    
        def backward(self, ids):
            """
            Class-specific backpropagation
            Either way works:
            1. self.logits.backward(gradient=one_hot, retain_graph=True)
            2. (self.logits * one_hot).sum().backward(retain_graph=True)
            """
    
            one_hot = self._encode_one_hot(ids)
            self.logits.backward(gradient=one_hot, retain_graph=True)
    
        def generate(self):
            raise NotImplementedError
    
        def remove_hook(self):
            """
            Remove all the forward/backward hook functions
            """
            for handle in self.handlers:
                handle.remove()
    
    
    class GradCAM(_BaseWrapper):
        """
        "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
        https://arxiv.org/pdf/1610.02391.pdf
        Look at Figure 2 on page 4
        """
    
        def __init__(self, model, candidate_layers=None):
            super(GradCAM, self).__init__(model)
            self.fmap_pool = OrderedDict()
            self.grad_pool = OrderedDict()
            self.candidate_layers = candidate_layers  # list
    
            def forward_hook(key):
                def forward_hook_(module, input, output):
                    # Save featuremaps
                    self.fmap_pool[key] = output.detach()
    
                return forward_hook_
    
            def backward_hook(key):
                def backward_hook_(module, grad_in, grad_out):
                    # Save the gradients correspond to the featuremaps
                    self.grad_pool[key] = grad_out[0].detach()
    
                return backward_hook_
    
            # If any candidates are not specified, the hook is registered to all the layers.
            for name, module in self.model.named_modules():
                if self.candidate_layers is None or name in self.candidate_layers:
                    self.handlers.append(module.register_forward_hook(forward_hook(name)))
                    self.handlers.append(module.register_backward_hook(backward_hook(name)))
    
        def _find(self, pool, target_layer):
            if target_layer in pool.keys():
                return pool[target_layer]
            else:
                raise ValueError("Invalid layer name: {}".format(target_layer))
    
        def _compute_grad_weights(self, grads):
            return F.adaptive_avg_pool2d(grads, 1)
    
        def forward(self, image):
            self.image_shape = image.shape[2:]
            return super(GradCAM, self).forward(image)
    
        def generate(self, target_layer):
            fmaps = self._find(self.fmap_pool, target_layer)
            grads = self._find(self.grad_pool, target_layer)
            weights = self._compute_grad_weights(grads)
    
            gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)
    
            gcam = F.interpolate(
                gcam, self.image_shape, mode="bilinear", align_corners=False
            )
    
            B, C, H, W = gcam.shape
            gcam = gcam.view(B, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam /= gcam.max(dim=1, keepdim=True)[0]
            gcam = gcam.view(B, C, H, W)
    
            return gcam 
        
    def demo2(image, label, model):
        """
        Generate Grad-CAM
        """
        # Model
        model = model
        model.to(device)
        model.eval()
    
        # The layers
        target_layers = ["conv2"]
        target_class = label
    
        # Images
        images = image.unsqueeze(0)
        gcam = GradCAM(model=model)
        probs, ids = gcam.forward(images)
        ids_ = torch.LongTensor([[target_class]] * len(images)).to(device)
        gcam.backward(ids=ids_)
    
        for target_layer in target_layers:
            print("Generating Grad-CAM @{}".format(target_layer))
    
            # Grad-CAM
            regions = gcam.generate(target_layer=target_layer)
            for j in range(len(images)):
                print(
                    "\t#{}: {} ({:.5f})".format(
                        j, classes[target_class], float(probs[ids == target_class])
                    )
                )
                
                gcam=regions[j, 0]
                plt.imshow(gcam.cpu())
                plt.show()
                
    image, label = next(iter(test_data_loader))
    # Load the model
    model = net
    # Grad cam
    demo2(image[0].to(device), label[0].to(device), model)
    
    
    image = np.transpose(image[0], (1,2,0))
    image2  = np.add(np.multiply(image.numpy(), np.array(norm_std)) ,np.array(norm_mean))
    print("True Class: ", classes[label[0].cpu()])
    plt.imshow(image)
    plt.show()
    plt.imshow(image2)
    plt.show()