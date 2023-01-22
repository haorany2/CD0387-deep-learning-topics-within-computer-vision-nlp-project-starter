import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from torch.utils.data import DataLoader
from PIL import Image, TiffImagePlugin
import os
import logging
from PIL import ImageFile
import sys
import subprocess
import json
import io
subprocess.check_call([sys.executable, "-m", "pip", "install", "nvgpu"])

ImageFile.LOAD_TRUNCATED_IMAGES = True
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
    
    
def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.inception_v3(pretrained=True)
    model.aux_logits = False

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(), 
        nn.Linear(256, 133)
    )
    with open(os.path.join(model_dir, "model.pth"), 'rb') as f:
        checkpoint = torch.load(f)
        model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    logger.info('Done loading model')
    return model

def load_from_bytearray(request_body):
    image_as_bytes = io.BytesIO(request_body)
    image = Image.open(image_as_bytes)
    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    #image_tensor = ToTensor()(image).unsqueeze(0)
    return test_transform(image).unsqueeze(0)

def input_fn(request_body, request_content_type):
    # if set content_type as "image/jpg" or "application/x-npy",
    # the input is also a python bytearray
    print('request_content_type', request_content_type)
    if request_content_type == "image/jpeg":
        image_tensor = load_from_bytearray(request_body)
    else:
        print("not support this type yet")
        
        raise ValueError("not support this type yet")
    return image_tensor

def predict_fn(input_data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.to(device)
    
    with torch.no_grad():
        model.eval()
        pred = model(input_data)
        pred=pred.argmax(dim=1, keepdim=True)
        print(pred)
    return pred

# # Serialize the prediction result into the desired response content type
# def output_fn(predictions, response_content_type):
#     return json.dumps(predictions)
