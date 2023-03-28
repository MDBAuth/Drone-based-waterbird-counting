import gc
import copy
import yaml
import torch
import numpy as np
import torchvision

from tqdm import tqdm
from pathlib import Path
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from utils.train_utils import collate_fn, get_transform, BirdDataset
from utils.utils import ensure_path

def train_loop_fn_old(data_loader, model, optimizer, device, scheduler=None, verbose=False):
    running_loss = 0.0
    problem_filenames = []
    for images, labels, filenames in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        optimizer.zero_grad()
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
    train_loss = running_loss / float(len(data_loader))
    scheduler.step(train_loss)
    return train_loss

def train_loop_fn(data_loader, model, optimizer, device, scheduler=None):
    """
    This is the "training loop", which calculates the loss for each epoch
    """
    running_loss = 0.0
    for images, labels, filenames in tqdm(data_loader, desc="Training"):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        # print(labels)
        optimizer.zero_grad()
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        running_loss += losses.item()
        del images, labels
        gc.collect()
        torch.cuda.empty_cache()
    train_loss = running_loss / float(len(data_loader))
    scheduler.step(train_loss)
    return train_loss

def eval_loop_fn_old(data_loader, model, device, verbose=False):
    running_loss = 0.0
    problem_filenames = []
    for images, labels, filenames in tqdm(data_loader, desc="Validation"):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        del images , labels
        gc.collect()
        torch.cuda.empty_cache()
    valid_loss = running_loss / float(len(data_loader))
    return valid_loss

def eval_loop_fn(data_loader, model, device):
    """
    This is the "validation loop", which calculates the loss for each epoch
    """
    running_loss = 0.0
    for images, labels, filenames in tqdm(data_loader, desc="Validation"):
        images = list(image.to(device) for image in images)
        labels = [{k: v.to(device) for k, v in l.items()} for l in labels]
        loss_dict = model(images, labels)
        losses = sum(loss for loss in loss_dict.values())
        running_loss += losses.item()
        del images , labels
        gc.collect()
        torch.cuda.empty_cache()
    valid_loss = running_loss / float(len(data_loader))
    return valid_loss

def train_mdba_old(params):
    lr = params['training']['learning_rate']
    wd = params['training']['weight_decay']
    momentum = params['training']['momentum']
    
    my_dataset = BirdDataset(
        root=params['training']['train_data_dir'],
        annotation=params['training']['train_coco'],
        transforms=get_transform())

    val_ratio = params['training']['val_ratio']
    total_size = len(my_dataset)
    val_size = int(val_ratio*total_size)
    train_size = total_size-val_size

    train_dataset, valid_dataset = torch.utils.data.random_split(
        my_dataset, (train_size, val_size))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['training']['train_batch_size'],
        shuffle=params['training']['train_shuffle_dl'],
        num_workers=params['training']['num_workers_dl'],
        collate_fn=collate_fn)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=params['training']['valid_batch_size'],
        shuffle=params['training']['valid_shuffle_dl'],
        num_workers=params['training']['num_workers_dl'],
        collate_fn=collate_fn)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.to(device)
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    weights_dir = ensure_path(params['models']['weights_dir']) 
    best_model_path = Path(params['models']['best_model'])
    early_stopping_patience = params['training']['early_stopping_patience']
    num_epochs = params['training']['num_epochs']
    all_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch --> {epoch+1} / {num_epochs}")
        print(f"-------------------------------")
        train_loss = train_loop_fn(train_loader, model, optimizer, device, scheduler)
        print('training Loss: {:.4f}'.format(train_loss))
        valid_loss = eval_loop_fn(valid_loader, model, device)
        print('validation Loss: {:.4f}'.format(valid_loss))
        all_losses.append(valid_loss)
        
        # torch.save(model, f"{weights_dir}/epoch-{epoch}_lr-{lr}.pth")
        if epoch == 0:
            torch.save(model, best_model_path)
            best_loss = valid_loss
            best_epoch = epoch
        elif valid_loss < best_loss:
            torch.save(model, best_model_path)
            best_loss = valid_loss
            best_epoch = epoch
        elif epoch - best_epoch == early_stopping_patience:
            print(f"Early stopping condition reached. Stopping training.")
            break
            
def train_mdba(params):
    """
    This is the main function that actually performs the training.

    Inputs: params from params.yaml file

    Outputs: a best model is saved in the best_model_path
    """

    # Define training parameters, typically no need to change these
    lr = params['training']['learning_rate']
    wd = params['training']['weight_decay']
    momentum = params['training']['momentum']
    early_stopping_patience = params['training']['early_stopping_patience']
    num_epochs = params['training']['num_epochs']

    best_model_path = Path(params['models']['best_model']) # Where output model will be saved
    ensure_path(params['models']['weights_dir']) # Ensure the output model folder exists

    # Define the dataset using training transforms (see ./utils/train_utils.py)
    # NOTE: it may be useful to split train and validation transforms before this step
    my_dataset = BirdDataset(
        root=params['training']['train_data_dir'],
        annotation=params['training']['train_coco'],
        transforms=get_transform())

    # Split the dataset into training and validation sets
    val_ratio = params['training']['val_ratio']
    total_size = len(my_dataset)
    val_size = int(val_ratio*total_size)
    train_size = total_size-val_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        my_dataset, (train_size, val_size))

    # Define data loaders for training and validation sets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=params['training']['train_batch_size'],
        shuffle=True,
        num_workers=params['training']['num_workers_dl'],
        collate_fn=collate_fn)

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=params['training']['valid_batch_size'],
        shuffle=False,
        num_workers=params['training']['num_workers_dl'],
        collate_fn=collate_fn)

    # Load the model into memory
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.to(device)

    # Define optimizer and scheduler
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(model_params, lr=lr, momentum=momentum, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Run main training loop, save a model when an epoch outperforms previous best
    all_losses = []
    for epoch in range(num_epochs):
        print(f"Epoch --> {epoch+1} / {num_epochs}")
        print(f"-------------------------------")
        train_loss = train_loop_fn(train_loader, model, optimizer, device, scheduler)
        print('training Loss: {:.4f}'.format(train_loss))
        valid_loss = eval_loop_fn(valid_loader, model, device)
        print('validation Loss: {:.4f}'.format(valid_loss))
        all_losses.append(valid_loss)
        
        ## Uncomment this line to save model after every epoch
        # torch.save(model, f"{weights_dir}/epoch-{epoch}_lr-{lr}.pth")

        if epoch == 0: # Save model as "best model" after first epoch
            torch.save(model, best_model_path)
            best_loss = valid_loss
            best_epoch = epoch
        elif valid_loss < best_loss: # Save model as "best model" if it results in lower loss
            torch.save(model, best_model_path)
            best_loss = valid_loss
            best_epoch = epoch
        elif epoch - best_epoch == early_stopping_patience: # Stop if no progress has been made
            print(f"Early stopping condition reached. Stopping training.")
            break
            
if __name__ == "__main__":
    with open("./params.yaml", "r") as params_file:
        params = yaml.safe_load(params_file)
    train_mdba(params)