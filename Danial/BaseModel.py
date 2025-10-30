import os
import torch
os.environ['TORCH_HOME'] = r'D:\23b6034\FYP\models'
def load():
    return torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
