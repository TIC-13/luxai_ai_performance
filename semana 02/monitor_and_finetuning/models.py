import torch
import torch.nn as nn

from utils import logging

from torchvision.models.densenet import densenet121, densenet161, densenet201

from torchvision.models.efficientnet import efficientnet_b0

from torchvision.models.resnet import resnet101, resnet152

AVAILABLE_MODELS = ["convnext_tiny", "convnext_small", 
    "convnext_base", "convnext_large", "convnext_xlarge", 
    "densenet121", "densenet161", "densenet201",
    "efficientnet_b0",
    "inception_v3",
    "resnet101", "resnet152"]


def show_available_models():
    return AVAILABLE_MODELS


def get_models(model_name, input_size, output_size, 
               pretrained=False, weightfile=None, train_all_layers=False):

    assert model_name in AVAILABLE_MODELS, f"{model_name} is not available. Provide a valid model ({AVAILABLE_MODELS});"

    model = None

    if "densenet" in model_name:

        pretrained_weights = None
        if pretrained:
            pretrained_weights = f"DenseNet{model_name[-3:]}_Weights.IMAGENET1K_V1"
        model = eval(model_name)(weights=pretrained_weights)
        prev_features_output = model.classifier.weight.shape[1]
        model.classifier = nn.Linear(prev_features_output, output_size)

        if not train_all_layers:
            logging("Training only Classifier Layer")
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            logging("Training all model")
        
        if weightfile:
            logging(f'Loading weight file from {weightfile}')
            model.load_state_dict(torch.load(weightfile))
    
    if "efficientnet" in model_name:

        pretrained_weights = None
        if pretrained:
            pretrained_weights = "EfficientNet_B0_Weights.IMAGENET1K_V1"
        model = eval(model_name)(weights=pretrained_weights)
        prev_features_output = model.classifier[1].weight.shape[1]
        model.classifier = nn.Linear(prev_features_output, output_size)

        if not train_all_layers:
            logging("Training only Classifier Layer")
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            logging("Training all model")

        if weightfile:
            logging(f"Loading weight file from {weightfile}")
            model.load_state_dict(torch.load(weightfile))
    if "resnet" in model_name:

        pretrained_weights = None
        if pretrained:
            pretrained_weights = f"ResNet{model_name[-3:]}_Weights.IMAGENET1K_V1"
        model = eval(model_name)(weights=pretrained_weights)
        prev_features_output = model.fc.weight.shape[1]
        model.fc = nn.Linear(prev_features_output, output_size)

        if not train_all_layers:
            logging("Training only Classifier Layer")
            for param in model.parameters():
                param.requires_grad = False
            
            for param in model.fc.parameters():
                param.requires_grad = True
        else:
            logging("Training all model")

        if weightfile:
            logging(f"Loading weight file from {weightfile}")
            model.load_state_dict(torch.load(weightfile))
    
    return model