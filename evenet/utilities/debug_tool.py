import torch.nn as nn
from collections import defaultdict
import torch

class DebugHookManager:
    def __init__(self, track_forward=True, track_backward=True, save_values=False):
        self.forward_hooks = []
        self.backward_hooks = []
        self.grad_hooks = []
        self.save_values = save_values

        self.forward_outputs = defaultdict(list)
        self.backward_grads = defaultdict(list)

        self.track_forward = track_forward
        self.track_backward = track_backward

    def check_forward(self, name):
        def hook(module, input, output):
            # Check output
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any():
                    print(f"[NaN Detected] Forward output of {name}")
                if torch.isinf(output).any():
                    print(f"[Inf Detected] Forward output of {name}")
                if self.save_values:
                    self.forward_outputs[name].append(output.detach().cpu())
            # Check inputs
            for i, inp in enumerate(input):
                if isinstance(inp, torch.Tensor):
                    if torch.isnan(inp).any():
                        print(f"[NaN Detected] Forward input {i} of {name}")
                    if torch.isinf(inp).any():
                        print(f"[Inf Detected] Forward input {i} of {name}")

        return hook

    def check_backward(self, name):
        def hook(module, grad_input, grad_output):
            for i, g in enumerate(grad_input):
                if isinstance(g, torch.Tensor):
                    if torch.isnan(g).any():
                        print(f"[NaN Detected] Grad input {i} of {name}")
                    if torch.isinf(g).any():
                        print(f"[Inf Detected] Grad input {i} of {name}")
                    if self.save_values:
                        self.backward_grads[name].append(g.detach().cpu())

        return hook

    def check_param_grad(self, name, param):
        def hook(grad):
            if grad is not None:
                if torch.isnan(grad).any():
                    print(f"[NaN Detected] Grad of param {name}")
                if torch.isinf(grad).any():
                    print(f"[Inf Detected] Grad of param {name}")
                if self.save_values:
                    self.backward_grads[f"param::{name}"].append(grad.detach().cpu())

        return hook

    def attach_hooks(self, model: nn.Module):
        for name, module in model.named_modules():
            if isinstance(module, nn.Module) and len(list(module.children())) == 0:  # only leaf modules
                if self.track_forward:
                    fh = module.register_forward_hook(self.check_forward(name))
                    self.forward_hooks.append(fh)
                if self.track_backward:
                    bh = module.register_full_backward_hook(self.check_backward(name))
                    self.backward_hooks.append(bh)

        for name, param in model.named_parameters():
            if param.requires_grad:
                gh = param.register_hook(self.check_param_grad(name, param))
                self.grad_hooks.append(gh)

    def remove_hooks(self):
        for h in self.forward_hooks + self.backward_hooks:
            h.remove()
        self.forward_hooks.clear()
        self.backward_hooks.clear()
        self.grad_hooks.clear()
        print("✅ All hooks removed.")

    def dump_debug_data(self):
        # Optional utility: Save collected outputs/grads to disk or analyze
        print("🔍 Dumped forward activations:")
        for k, v in self.forward_outputs.items():
            print(f"{k}: {len(v)} tensors")

        print("🔍 Dumped backward gradients:")
        for k, v in self.backward_grads.items():
            print(f"{k}: {len(v)} tensors")
