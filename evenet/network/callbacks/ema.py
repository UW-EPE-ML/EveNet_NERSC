import lightning.pytorch as pl


class EMACallback(pl.Callback):
    def __init__(self, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.ema_state = {}
        self.initialized = False
        self.model_parts = None

    def on_train_start(self, trainer, pl_module):
        # Get model_parts from the pl_module now that it's initialized
        self.model_parts = pl_module.model_parts  # Access dynamically
        self._initialize_ema()

    def _initialize_ema(self):
        for name, modules in self.model_parts.items():
            for module in modules:
                for n, param in module.named_parameters():
                    if param.requires_grad:
                        self.ema_state[f"{name}.{n}"] = param.data.clone()
        self.initialized = True

    def on_after_backward(self, trainer, pl_module):
        if not self.initialized:
            return
        for name, modules in self.model_parts.items():
            for module in modules:
                for n, param in module.named_parameters():
                    if param.requires_grad:
                        key = f"{name}.{n}"
                        self.ema_state[key].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_ema_weights(self, pl_module):
        self.original_state = {}
        for name, modules in self.model_parts.items():
            for module in modules:
                for n, param in module.named_parameters():
                    if param.requires_grad:
                        key = f"{name}.{n}"
                        self.original_state[key] = param.data.clone()
                        param.data.copy_(self.ema_state[key])

    def restore_original_weights(self, pl_module):
        for name, modules in self.model_parts.items():
            for module in modules:
                for n, param in module.named_parameters():
                    if param.requires_grad:
                        key = f"{name}.{n}"
                        param.data.copy_(self.original_state[key])

    def on_validation_start(self, trainer, pl_module):
        if self.model_parts is None:
            self.model_parts = pl_module.model_parts
            self._initialize_ema()
        self.apply_ema_weights(pl_module)

    def on_validation_end(self, trainer, pl_module):
        self.restore_original_weights(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        self.apply_ema_weights(pl_module)

    def on_save_checkpoint_end(self, trainer, pl_module):
        self.restore_original_weights(pl_module)
