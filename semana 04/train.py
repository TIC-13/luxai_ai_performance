
import os
import math
import tqdm

import wandb
import torch

import argparse
import torchvision

from models import get_model
from utils import logging, makedir, get_current_time

from sklearn.metrics import accuracy_score, f1_score

(DEVICE, DEVICE_NAME) = ('cuda:0', torch.cuda.get_device_name()) if torch.cuda.is_available() else ('cpu', "CPU")


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_data', required=True)
    parser.add_argument('--validation_data', required=True)

    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--valid_step',  default=1, type=int)

    parser.add_argument('--epochs',     default=100, type=int)
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--batch_size', default=256,  type=int)
    
    parser.add_argument('--optimizer', default="SGD",  type=str)
    parser.add_argument('--lr', default=1e-3,  type=float)
    parser.add_argument('--weight_decay', default=1e-8, type=float)
    parser.add_argument('--beta1', default=0.9,  type=float)
    parser.add_argument('--beta2', default=0.99,  type=float)

    parser.add_argument('--model', default='mobilenet', choices=['densenet', 'mobilenet'])
    parser.add_argument('--input_size', default=224, type=int)
    
    parser.add_argument('--outputdir', default='./rsc/outputs/')

    parser.add_argument('--logfile', default='parameters.txt', type=str)

    return parser.parse_args()

def train_step(model, loader, optimizer, criterion, device='cpu', logger=None):
    
    model = model.to(device)
    model.train()

    train_loss = 0.
    targets = list()
    predictions = list()

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
        
        predictions = predictions + predicted.cpu().numpy().tolist()
        targets     = targets     + target.cpu().numpy().tolist()

    return train_loss, targets, predictions

def validation_step(model, loader, criterion, device='cpu'):

    model.to(device)
    model.eval()

    valid_loss    = 0.
    
    targets = list()
    predictions = list()

    with torch.no_grad():
        for (sample, target) in tqdm.tqdm(loader):
            sample = sample.to(device)
            target = target.to(device)
            output = model(sample)
            loss = criterion(output, target)

            valid_loss += loss.item()
            
            #_, labels    = torch.max(target, 1)
            _, predicted = torch.max(output, 1)

            predictions = predictions + predicted.cpu().numpy().tolist()
            targets     = targets     + target.cpu().numpy().tolist()

    return valid_loss, targets, predictions

def get_optimizer(name="Adam", params=None, **kwargs):
    if name == "SGD":
        return torch.optim.SGD(params, lr=kwargs["lr"])
    elif name == "Adam":
        return torch.optim.Adam(params, lr=kwargs["lr"], betas=(kwargs["beta1"], kwargs["beta2"]))
    elif name == "AdamW":
        return torch.optim.AdamW(params, lr=kwargs["lr"], weight_decay=kwargs["weight_decay"])
    else: 
        return None

def get_loss_function():
    return torch.nn.CrossEntropyLoss(reduction='mean')

def get_transforms(target_image_size=224):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(target_image_size+50),
        torchvision.transforms.CenterCrop(target_image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomRotation(20),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
    ])

    validation_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(target_image_size+50),
        torchvision.transforms.CenterCrop(target_image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
    ])

    return train_transforms, validation_transforms


if __name__ == '__main__':

    args = get_args()

    # LOGGER
    curr_time   = get_current_time()
    logdir = os.path.join(args.outputdir, curr_time)
    makedir(logdir)
    wandb.init(project="Torch Classification", name=logdir)

    wandb.run.summary["Batch Size"] = args.batch_size
    wandb.run.summary["Optimizer"]  = args.optimizer
    wandb.run.summary["Image size"]  = args.input_size
    wandb.run.summary["lr"]  = args.lr
    wandb.run.summary["decay"]  = args.weight_decay
    wandb.run.summary["model"]  = args.model

    # DATASET
    train_transforms, validation_transforms = get_transforms(args.input_size)

    train_dataset = torchvision.datasets.ImageFolder(root=args.train_data, transform=train_transforms)
    valid_dataset = torchvision.datasets.ImageFolder(root=args.validation_data, transform=validation_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    classes     = os.listdir(args.train_data)
    valid_classes = os.listdir(args.validation_data)

    assert classes == valid_classes, "Different classes found at train and validation folders"

    # MODEL AND LOSS
    num_classes = len(classes)
    model = get_model(args.model, num_classes, fine_tune_classifier=False)

    criterion = get_loss_function()
    optimizer = get_optimizer(args.optimizer, model.parameters(), lr=args.lr, 
                              beta1=args.beta1, beta2=args.beta2, weight_decay=args.weight_decay)

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
    logging(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Optimizer: {args.optimizer} | Initial LR: {args.lr} | DECAY: {args.weight_decay}\n")

    for epoch in range(args.epochs):
        
        logging(f"(Train step - Epoch {epoch}) | Current LR: {scheduler.get_last_lr()[0]}")
        
        train_loss, train_targets, train_predictions = train_step(model, train_loader, optimizer, criterion, device=DEVICE)
        train_acc = accuracy_score(train_targets, train_predictions)
        train_f1  = f1_score(train_targets, train_predictions)
        train_loss = math.sqrt(train_loss/args.batch_size)
        wandb.log({"train loss": train_loss, "train_acc": train_acc, "train_f1": train_f1})

        logging(f"Train loss: {train_loss}, Train acc: {train_acc}")
        
        if epoch % args.valid_step == 0:
            
            valid_loss, valid_targets, valid_predictions = validation_step(model, valid_loader, criterion, device=DEVICE)
            valid_acc = accuracy_score(valid_targets, valid_predictions)
            valid_f1  = f1_score(valid_targets, valid_predictions)
            valid_loss = math.sqrt(valid_loss/args.batch_size)
            
            logging(f"Val. loss: {valid_loss}, Val. acc: {valid_acc}")
            wandb.log({"valid loss": valid_loss, "valid_acc": valid_acc, "valid_f1": valid_f1})

            if valid_acc > best_acc:
                logging(f"Best model found at epoch {epoch}; Saving weights...")
                best_acc = valid_acc
                best_epoch = epoch
                last_update_best_acc = 0
                best_acc_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(logdir, "model.pth"))
            else:
                last_update_best_acc = last_update_best_acc + 1
            
        if train_acc > 99.9 or (last_update_best_acc > 20 and valid_loss > best_acc_loss):
            logging(f"Stopping early at epoch {epoch}")
            break
        
        scheduler.step()
    
    logging(f"Last train ACC: {train_acc}| Last Validation ACC: {valid_acc}") 
    logging(f"Best ACC ({best_acc}) found at epoch {best_epoch}")
        
