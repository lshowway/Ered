import torch


# model_weights = torch.load('G:\D\MSRA\knowledge_aware\checkpoints\luke_large_500k/pytorch_model.bin', map_location='cpu')
#
# new_dict = {}
# for k, v in model_weights.items():
#     k = 'luke.'+k
#     new_dict[k] = v

# torch.save(new_dict, 'G:\D\MSRA\knowledge_aware\checkpoints\luke_large_500k/pytorch_model_2.bin')


weights = torch.load('G:\D\MSRA\knowledge_aware\checkpoints\luke_large_500k/pytorch_model_2.bin', map_location='cpu')

for k, v in weights.items():
    print(k)
