import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import LlamaForCausalLM
from models.modeling_llama import BinaryMoSForCausalLM
from binarization.binarized_modules import BinaryMoSLinear

def get_blocks(model):
    if model.__class__.__name__ == 'LlamaForCausalLM':
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers
            
def replace_modules(root_module, num_expert=4, do_train=False, print_layers=False):
    module_name_dict = {name: module for name, module in root_module.named_modules()}
    for name, module in module_name_dict.items():
        if isinstance(module, nn.Linear):
            ind = name.rfind(".")
            if ind == -1:
                father = module_name_dict[""]
            else:
                father = module_name_dict[name[:ind]]
            mos_linear = BinaryMoSLinear(module.weight, module.bias, num_expert, do_train)
            setattr(father, name[ind + 1 :], mos_linear)
            if print_layers:
                print(f"replace layer {name} with {mos_linear}") 