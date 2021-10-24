import torch

# 현재 Setup 되어있는 device 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Number of available devices:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))
