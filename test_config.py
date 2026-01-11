import torch
path = "/share/project/huangxu/models/SAE/models/dis.pth"
s = torch.load(path)
print(s)