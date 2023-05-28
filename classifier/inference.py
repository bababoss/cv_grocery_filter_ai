import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import os
import copy

import numpy as np
import cv2
import PIL.Image as Image


import warnings
warnings.filterwarnings('ignore')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
classes = ['lemon', 'tometo']





    
def image_tensor(cv2_image):
    # Read a PIL image

    # Convert BGR image to RGB image
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Define a transform to convert PIL 
    # image to a Torch tensor
    transform = transforms.Compose([
          transforms.Resize((256,256)),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    img_tensor = transform(image)

    # print the converted Torch tensor

    return img_tensor



def predict(cv2_image, model):
    img=image_tensor(cv2_image)
    pred_img = img.view(1,3,256,256).to(device)
    pred = torch.max(model(pred_img),1)[1]
    return classes[pred].capitalize()
    