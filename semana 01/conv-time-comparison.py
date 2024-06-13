
import torch
import cv2
import numpy as np
import time

from convolution import NumpyConvolution2D

import torch.nn.functional as F

# Carregando e processando entrada
image = cv2.imread('input.png')
image = np.moveaxis(image, 2, 0)    # Ajustando dimensões (H x W x C -> C x H x W) 
image = image / 255.                # Normalizando [0 , 1]

torch_image = torch.Tensor(image)

input_tensor = torch.unsqueeze(torch_image, 0) # Ajustando formato para o PyTorch (B, C, H, W)
input_array  = image

# Definindo kernel 
weight = np.array([[[[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], 
                    [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]]])

print(weight.shape)

# Parametros da convolução 
channels_in  = 3
channels_out = 3
kernel_size  = 3
stride       = 1
padding      = 1

cn = NumpyConvolution2D(3,3,3,weight=weight)

start_numpy = time.time()
out = cn.forward(input_array)
time_numpy   = time.time() - start_numpy

start_torch = time.time()
F.conv2d(input=input_tensor.cpu(), weight=torch.Tensor(weight).cpu(), stride=stride, padding=padding)
time_torch  = time.time() - start_torch

print(f"NUMPY 2D: {time_numpy}\nTORCH 2D: {time_torch}")
