import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install('smdebug')
install('keras')
install('tensorflow')


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

import os
import json

import keras 
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import smdebug.pytorch as smd

from tensorflow.keras.optimizers import SGD

import argparse
import json
import logging
import os
import sys
import boto3

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


from s3fs.core import S3FileSystem
s3 = S3FileSystem()


def train(model, train_loader, criterion, optimizer, epoch, hook):
    model.train()
    hook.set_mode(smd.modes.TRAIN)

    for epoch in range(1, epoch + 1):
      
        #for batch_idx, (data, target) in enumerate(train_loader, 1):
        for batch_idx, sample in enumerate(train_loader):
            
            idx=batch_idx+1
            
            data=sample[0]
            target=sample[1]
            
            optimizer.zero_grad()
            output = model(data.permute(0, 3, 1, 2))
            label=np.array([np.argmax(i) for i in target])
            loss = F.nll_loss(output, torch.from_numpy(label))
            loss.backward()
            optimizer.step()
            #if batch_idx % 100 == 0:
            logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        idx * len(data),
                        len(train_loader.dataset),
                        100.0 * idx / len(train_loader),
                        loss.item(),
                    )
                )
            
def test(model, test_loader, hook):
    model.eval()
    hook.set_mode(smd.modes.EVAL)

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.permute(0, 3, 1, 2))
            label=np.array([np.argmax(i) for i in target])
            test_loss += F.nll_loss(output, torch.from_numpy(label), size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(torch.from_numpy(label).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )
    
def net():
    model = models.resnet18()    

    for param in model.parameters():
        param.requires_grad = False        

    num_features=model.fc.in_features          
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)   


def main(args):
    
    bucket= 'udacity3-project'
    train_x = np.load(s3.open('{}/{}'.format(bucket, 'dog-breeding/train_x.npy')))
    train_y = np.load(s3.open('{}/{}'.format(bucket, 'dog-breeding/train_y.npy')))
    test_x = np.load(s3.open('{}/{}'.format(bucket, 'dog-breeding/test_x.npy')))
    test_y = np.load(s3.open('{}/{}'.format(bucket, 'dog-breeding/test_y.npy')))
    
    train_x=torch.from_numpy(train_x)
    train_y=torch.from_numpy(train_y)
    test_x=torch.from_numpy(test_x)
    test_y=torch.from_numpy(test_y)
    
    train_dataset = torch.utils.data.TensorDataset(train_x,train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x,test_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0,)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=True,num_workers=0,)
    
    model=net()
    
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    epoch=args.epochs
    
    train(model, train_loader, criterion, optimizer, epoch,hook)
    
    test(model, test_loader,hook)
    
    save_model(model, args.model_dir)

    torch.save(model.state_dict(), "dog_train.pt")
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        metavar="N",
        help="input batch size for training (default: 100)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=50,
        metavar="N",
        help="input batch size for testing (default: 50)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="N",
        help="number of epochs to train (default: 2)",
    )

    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    
    #parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
    #parser.add_argument("--train-dir", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument("--test-dir", type=str, default=os.environ['SM_CHANNEL_TEST'])

    args, unknown = parser.parse_known_args()
    
    #print(args.train_dir)
    #print(args.test_dir)
    
    main(args)
