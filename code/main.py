import torch
import torch.nn as nn
from functions.model.deeplab import DeepLabv3_plus, get_1x_lr_params, get_10x_lr_params


model = DeepLabv3_plus(n_classes=3, os=16, nInputChannels = 1)

train_params = [{'params': get_1x_lr_params(model), 'lr': 0.005},
                {'params': get_10x_lr_params(model), 'lr': 0.05}]

# Define Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(train_params)










