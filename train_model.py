#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
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
import boto3
from PIL import ImageFile
import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "smdebug"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-lightning"])

import smdebug.pytorch as smd
from smdebug import modes
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    hook.set_mode(modes.EVAL)
    correct = 0
    running_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            #NOTE: Notice how we are not changing the data shape here
            # This is because CNNs expects a 3 dimensional input
            data = data.to(device)
            target = target.to(device)
            pred = model(data)
            loss = criterion(pred, target)
            pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            
            running_loss += loss.item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    logger.info(f"Test set: Average loss: {running_loss/len(test_loader.dataset)}")
    logger.info(f'Test set: Accuracy: {correct}/{len(test_loader.dataset)} = {100*(correct/len(test_loader.dataset))}%)')



def train(model, train_loader, valid_loader, criterion, optimizer, device, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    for e in range(args.epochs):
        running_loss = 0
        correct = 0
        valid_loss = 0
        valid_correct = 0
        model.train()
        hook.set_mode(modes.TRAIN)
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred = model(data)
            loss = criterion(pred, target)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            pred=pred.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        logger.info(f"Epoch {e}: Training Loss {running_loss/len(train_loader.dataset)}, Training Accuracy {100*(correct/len(train_loader.dataset))}%")
            
        model.eval()
        hook.set_mode(modes.EVAL)
        with torch.no_grad():
            for (data, target) in valid_loader:
                data = data.to(device)
                target = target.to(device)
                pred = model(data)
                loss = criterion(pred, target)
                valid_loss += loss.item()
                pred=pred.argmax(dim=1, keepdim=True)
                valid_correct += pred.eq(target.view_as(pred)).sum().item()
        logger.info(f"Epoch {e}: Val Loss {valid_loss/len(valid_loader.dataset)}, Val Accuracy {100*(valid_correct/len(valid_loader.dataset))}%")       
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
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
    return model


def create_data_loaders(data, path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_transform = transforms.Compose([
    transforms.RandomResizedCrop(299),
    transforms.RandomRotation(degrees=60),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])])

    test_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    if data == 'train':
        dataset = datasets.ImageFolder(root=path, transform = train_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    elif data == 'valid' or data == 'test':
        dataset = datasets.ImageFolder(root=path, transform = test_transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader, dataset.classes


def main(args):
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")

    model=net()
    model=model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum= args.momentum)
    
    '''
    TODO: Create hook
    '''
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    hook.register_loss(loss_criterion)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    
    
    train_loader, class_dict = create_data_loaders('train', args.train_dataset_dir, args.batch_size)
    valid_loader, _ = create_data_loaders('valid', args.valid_dataset_dir, args.test_batch_size)
    
    model=train(model, train_loader, valid_loader, loss_criterion, optimizer, device, hook)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test_loader, _ = create_data_loaders('test', args.test_dataset_dir, args.test_batch_size)
    test(model, test_loader, loss_criterion, device, hook)
    
    '''
    TODO: Save the trained model
    '''
    torch.save({
            'epoch': args.epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'output_class_dict': class_dict
            }, os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.8, metavar="M", help="SGD momentum (default: 0.8)"
    )
    
    parser.add_argument('--train-dataset-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--valid-dataset-dir', type=str, default=os.environ.get('SM_CHANNEL_VALID'))
    parser.add_argument('--test-dataset-dir', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    args=parser.parse_args()
    
    
    
    main(args)
