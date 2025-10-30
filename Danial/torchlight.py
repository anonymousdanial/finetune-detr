import torch

def transfer():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device