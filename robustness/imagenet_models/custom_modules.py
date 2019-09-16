import torch 
from torch import nn
ch = torch

class FakeReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class FakeReLUM(nn.Module):
    def forward(self, x):
        return FakeReLU.apply(x)

class SequentialWithArgs(torch.nn.Sequential):
    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l-1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input

class PushupReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        inp, = ctx.saved_tensors
        # Replace with zeros where input and gradient is negative
        need_pushup = ch.where((inp < 0) & (grad_output < 0),
                ch.zeros_like(inp), grad_output)
        
        # Replace the zero gradients with the pushup ones
        final_grad = ch.where(inp < 0, need_pushup, grad_output)
        
        return final_grad

class PushupReLUM(nn.Module):
    def __init__(self, *args, **kwargs):
        super(PushupReLUM, self).__init__()
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        if self.counter > 16:
            return nn.functional.relu(x)

        return PushupReLU.apply(x)
