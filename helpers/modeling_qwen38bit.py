import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM
from module import QuantizedLinear

class MyQwenForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        for name, module in self.named_modules():
            if isinstance(module, QuantizedLinear):
                continue
            for child_name, child in module.named_children():
                if child_name=="lm_head":
                    continue
                if isinstance(child, torch.nn.Linear):
                    ql = QuantizedLinear(child.in_features, child.out_features, bias=(child.bias is not None))
                    setattr(module, child_name, ql)
