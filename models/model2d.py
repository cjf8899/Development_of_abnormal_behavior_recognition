from __future__ import division
import torch
from torch import nn
import pdb
import torchvision.models as models

def generate_model( opt):
    assert opt.model in ['resnext', 'resnet']
    assert opt.model_depth in [50, 101]

    if opt.model == 'resnext':
        assert opt.model_depth in [50]
        print("resnext50_32x4d ....")
        model = models.resnext50_32x4d(pretrained=True)
    
    elif opt.model == 'resnet':
        if opt.model_depth == 50:
            print("resnet50 ....")
            model = models.resnet50(pretrained=True)
        else:
            print("resnet101 ....")
            model = models.resnet101(pretrained=True)
            
    model.fc = nn.Linear(model.fc.in_features, opt.n_finetune_classes)
    model.fc = model.fc.cuda()
    model = model.cuda()
    model = nn.DataParallel(model)
    
    return model, model.parameters()

