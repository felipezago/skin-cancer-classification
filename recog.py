# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 09:44:17 2020

@author: felip
"""
import torch
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path
import torchvision
import torch.nn as nn


device = torch.device('cpu')

classes = [ 'carcinoma basocelular','ceratoses actínicas', 'lesoes de ceratose benignas', 
               'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']
num_classes = len(classes)
net = torchvision.models.resnet18(pretrained = True)
net.fc = nn.Linear(512, num_classes)
net = net.to(device)

PATH = 'arquivos/model.pth'

model = net.load_state_dict(torch.load(PATH, map_location=device))
net = net.eval()
trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ])

image = Image.open(Path('arquivos/classes/df/ISIC_0024318.jpg'))

input = trans(image)

input = input.view(1, 3, 32,32)

output = net(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(classes[prediction])