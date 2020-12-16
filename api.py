# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 22:55:20 2020

@author: felip
"""
import io
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from flask import Flask, jsonify, request
import torchvision

device = torch.device('cpu')

app = Flask(__name__)
classes = [ 'carcinoma basocelular','ceratoses actínicas', 'lesoes de ceratose benignas', 
               'dermatofibroma','melanoma', 'nevos melanocíticos', 'lesões vasculares']
num_classes = len(classes)
net = torchvision.models.resnet18(pretrained = True)
net = net.to(device)
net.fc = nn.Linear(512, num_classes)



PATH = 'arquivos/model.pth'
#model = net.load_state_dict(torch.load(PATH, map_location=device))


@app.route('/')
def hello():
    return "hello world"

def transform_image(image_bytes):
    my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                            [0.485, 0.456, 0.406],
                                            [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    return my_transforms(image).unsqueeze(0)

def get_prediction(image_bytes):
    
    net.fc = nn.Linear(512, num_classes)
    net.load_state_dict(torch.load(PATH))
    tensor = transform_image(image_bytes=image_bytes)
    outputs = net(tensor)
    print(outputs)
    _, predicted = torch.max(outputs.data, 1)
    print(predicted)
    print(classes[predicted])
    return classes[predicted]


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        class_id, class_name = get_prediction(image_bytes=img_bytes)
        return jsonify({'class_id': class_id, 'class_name': class_name})

with open("arquivos/classes/df/ISIC_0024318.jpg", 'rb') as f:
    image_bytes = f.read()
    print(get_prediction(image_bytes=image_bytes))

if __name__ == '__main__':
    app.run()