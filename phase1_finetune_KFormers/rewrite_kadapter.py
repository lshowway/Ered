import torch

t = torch.load('G:\D\MSRA\knowledge_aware\checkpoints/fac-adapter\pytorch_model.bin', map_location='cpu')
for x in t.keys():
    print(x)