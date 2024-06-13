
import os

import torch
import math
import tqdm
import argparse

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils import logging, makedir, get_current_time

from models import get_models
from torch.utils.tensorboard import SummaryWriter
import wandb

(DEVICE, DEVICE_NAME) = ('cuda:0', torch.cuda.get_device_name()) if torch.cuda.is_available() else ('cpu', "CPU")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--valid_step',  default=1, type=int)

    parser.add_argument('--epochs',     default=100, type=int)
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--batch_size', default=256,  type=int)
    
    parser.add_argument('--lr', default=5e-5,  type=float)
    parser.add_argument('--decay', default=1e-8,  type=float)

    parser.add_argument('--model', default='densenet121', choices=['densenet121', 'densenet161', 'densenet201',
                                                                      'efficientnet_b0',
                                                                      'resnet101', 'resnet152'])
    parser.add_argument('--input_size', default=224, type=int)
    
    parser.add_argument('--weightfile', default=None, type=str)
    parser.add_argument('--outputdir', default='./rsc/outputs/')

    parser.add_argument('--pretrained',       action='store_true', default=False)
    parser.add_argument('--train_all_layers', action='store_true', default=False)

    parser.add_argument('--logfile', default='parameters.txt', type=str)

    return parser.parse_args()


def train_step(model, loader, optimizer, criterion, device='cpu', logger=None):
    
    model = model.to(device)
    model.train()

    train_loss = 0.
    total_samples = 0.
    total_correct = 0.

    for (sample, target) in tqdm.tqdm(loader):

        sample = sample.to(device)
        target = target.to(device)

        im_grid = torchvision.utils.make_grid(sample)
        if logger:
            logger.add_image('Samples', im_grid)

        optimizer.zero_grad()

        output = model(sample)

        loss = criterion(output, target)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        
        #_, labels    = torch.max(target, 1)
        _, predicted = torch.max(output, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += target.size(0)
    
    accuracy   = 100 * total_correct / total_samples

    return train_loss, accuracy


def validation_step(model, loader, criterion, device='cpu'):

    model.to(device)
    model.eval()

    valid_loss    = 0.
    total_samples = 0.
    total_correct = 0.

    with torch.no_grad():
        for (sample, target) in tqdm.tqdm(loader):
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            loss = criterion(output, target)

            valid_loss += loss.item()
            
            #_, labels    = torch.max(target, 1)
            _, predicted = torch.max(output, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += target.size(0)

    accuracy   = 100 * total_correct / total_samples

    return valid_loss, accuracy


if __name__ == '__main__':

    args = get_args()

    train_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.485,.456,.406],
                             std=[.229,.224,.225])
    ])

    curr_time   = get_current_time()
    output_path = "./outputs/tensorboard"
    logdir = os.path.join(output_path, curr_time)
    makedir(logdir)
    
    wandb.init(project="Torch Classification", name=logdir)

    train_dataset = ImageFolder(root="./../data/train", transform=train_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    valid_dataset = ImageFolder(root="./../data/valid", transform=valid_transforms)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    
    classes     = ["Jerry", "Tom"]
    num_classes = len(classes)
    model = get_models(args.model, args.input_size, num_classes, 
                       pretrained=args.pretrained, 
                       weightfile=args.weightfile, train_all_layers=args.train_all_layers)

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)

    def warmup(current_step: int):
        return 1 / (10 ** (float(args.warmup - current_step)))
    
    train_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)
    scheduler = train_scheduler

    train_losses  = list()
    valid_losses  = list()
    train_accs    = list()
    valid_accs    = list()
    best_acc      = -1.
    best_epoch    = 0.
    best_acc_loss = 1000.
    last_update_best_acc = 0

    logging(f"Starting training using device: {DEVICE_NAME}")
    logging(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Initial LR: {args.lr} | DECAY: {args.decay}\n")

    for epoch in range(args.epochs):
        
        logging(f"(Train step - Epoch {epoch}) | Current LR: {scheduler.get_last_lr()[0]}")
        train_loss, train_acc = train_step(model, train_loader, optimizer, criterion, device=DEVICE)
        train_loss = math.sqrt(train_loss/args.batch_size)
        wandb.log({"train loss": train_loss, "train_acc": train_acc})

        logging(f"Train loss: {train_loss}, Train acc: {train_acc}")
        
        
        if epoch % args.valid_step == 0:
            valid_loss, valid_acc = validation_step(model, valid_loader, criterion, device=DEVICE)
            valid_loss = math.sqrt(valid_loss/args.batch_size)
            
            logging(f"Val. loss: {valid_loss}, Val. acc: {valid_acc}")
            wandb.log({"valid loss": valid_loss, "valid_acc": valid_acc})

            if valid_acc > best_acc:
                logging(f"Best model found at epoch {epoch}; Saving weights...")
                best_acc = valid_acc
                best_epoch = epoch
                last_update_best_acc = 0
                best_acc_loss = valid_loss
            else:
                last_update_best_acc = last_update_best_acc + 1
            
        if train_acc > 99.9 or (last_update_best_acc > 20 and valid_loss > best_acc_loss):
            logging(f"Stopping early at epoch {epoch}")
            break
        
        scheduler.step()
    
    logging(f"Last train ACC: {train_acc}| Last Validation ACC: {valid_acc}") 
    logging(f"Best ACC ({best_acc}) found at epoch {best_epoch}")
        
