import torch
import torch.nn.functional as F

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(self, x, scale):
        return torch.round(x / scale).clamp(-128, 127) * scale
    
    @staticmethod
    def backward(self, g):
        return g, None


class QuantizedLinear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias:True):
        super().__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.register_buffer("scale", torch.tensor(1.0))

    def load_weights(self, module):
        self.linear.load_state_dict(module.state_dict())
        max_val = self.linear.weight.abs().max()
        if max_val != 0.0:
            self.scale.copy_(torch.clamp(max_val / 127, min=1e-6))

    def forward(self, x):
        w_q = FakeQuantize.apply(self.linear.weight, self.scale)
        return F.linear(x, w_q, self.linear.bias)
