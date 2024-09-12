import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import prod

class STEBinary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.sign()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        deriv = ((input > -1) & (input < 1))
        grad_output = grad_output * deriv
        return grad_output

class BinaryMoSLinear(nn.Module):
    def __init__(self, weight, bias, num_expert, do_train):
        super(BinaryMoSLinear, self).__init__()
        self.weight = nn.Parameter(weight.data)
        if bias is not None:
            self.bias = nn.Parameter(bias.data)
        else:
            self.bias = None
            
        self.out_channel_shape = self.weight.shape[0]
        self.in_channel_shape = self.weight.shape[1]
        self.hidden_dim = self.weight.shape[1]
        self.num_experts = num_expert
        self.do_train = do_train

        self.gate_linear = nn.Linear(self.hidden_dim, self.num_experts, bias=False, device=self.weight.device)
        if self.do_train:
            reduced_rank = 1
            U, S, Vh = torch.linalg.svd(abs(weight.data.clone().float()), full_matrices=False)
            out_channel_scale = (U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))).view(-1).repeat(self.num_experts, 1)
            in_channel_scale = (torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh).view(-1).repeat(self.num_experts, 1)
        else:
            in_channel_scale = torch.zeros(self.num_experts, self.weight.shape[1]).to(device=self.weight.device)
            out_channel_scale = torch.zeros(self.num_experts, self.weight.shape[0]).to(device=self.weight.device)

        self.register_parameter('in_channel_scale', nn.Parameter(in_channel_scale))
        self.register_parameter('out_channel_scale', nn.Parameter(out_channel_scale))
            

    def forward(self, x):
        *seqlen, hidden_dim = x.shape
        seqlen.append(self.out_channel_shape)
        final_hidden_output_dim = tuple(seqlen)
        x = x.view(-1, hidden_dim)

        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate_linear(x)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(x.dtype)
        
        in_scale_expert = routing_weights.matmul(self.in_channel_scale)
        out_scale_expert = routing_weights.matmul(self.out_channel_scale)
        
        if self.bias is not None:
            final_hidden_states = (((x * in_scale_expert) @ self.binarize().t()) * out_scale_expert) + self.bias
        else:
            final_hidden_states = (((x * in_scale_expert) @ self.binarize().t()) * out_scale_expert)
        final_hidden_states = final_hidden_states.reshape(final_hidden_output_dim)

        return final_hidden_states

    def binarize(self):
        binary_weight = STEBinary().apply(self.weight)

        return binary_weight
    

    def extra_repr(self):
        return f'in_features={self.weight.shape[1]}, out_features={self.weight.shape[0]}, bias={self.bias is not None}, num_experts={self.num_experts}'
