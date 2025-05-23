import copy
import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, device=None):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.device = device
        self._has_module = hasattr(self.ema_model, 'module')  # for DDP

        # Turn off gradients for ema_model
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module):
        ema_params = self._get_params(self.ema_model)
        model_params = self._get_params(model)

        for ema_p, model_p in zip(ema_params, model_params):
            if self.device is not None:
                model_p = model_p.to(self.device)
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def _get_params(self, model):
        return (
            model.module.parameters() if self._has_module else model.parameters()
        )

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict):
        self.ema_model.load_state_dict(state_dict)

    def to(self, device):
        self.device = device
        self.ema_model.to(device)

    def copy_to(self, model: nn.Module):
        """Copies EMA weights into the given model."""
        model_params = self._get_params(model)
        ema_params = self._get_params(self.ema_model)
        for m, e in zip(model_params, ema_params):
            m.data.copy_(e.data)

    def ema_model_eval(self):
        self.ema_model.eval()
        return self.ema_model
