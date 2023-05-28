import torch
from PIL import Image
import cv2
import torchvision.transforms as transforms

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import os
import copy
from detector import object_detector

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model():
    PATH="models/fruit_model.pth"
    
    model_conv = models.resnet18(pretrained=True)

    #freeze the pre-trained weights on convolutional layer
    for param in model_conv.parameters():
        param.requires_grad = False

    #classifier
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    #set the gpu
    model_conv = model_conv.to(device)
    model_conv.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
    model_conv.eval()
    
    return model_conv


model_net=load_model()

object_detector.run_detector(model_net,video_path=None)




