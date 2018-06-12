import torch
from torch.autograd import Variable


# def cast_torch_float(tensor, requires_grad=True):
#     return Variable(torch.from_numpy(tensor).float(), requires_grad=requires_grad)
#
#
# def cast_torch_int(tensor, requires_grad=True):
#     return Variable(torch.from_numpy(tensor).long(), requires_grad=requires_grad)

def cast_torch_float(tensor, requires_grad=True):
    return torch.from_numpy(tensor).float().requires_grad_(requires_grad)


def cast_torch_int(tensor, requires_grad=True):
    return torch.from_numpy(tensor).long().requires_grad_(requires_grad)
