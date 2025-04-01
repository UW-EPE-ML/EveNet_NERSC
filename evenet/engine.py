from typing import Any, Dict

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn


# Dummy model function
def model(x):
    return 1


class EveNetEngine(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = None

        print(f"{self.__class__.__name__} initialized")

    def forward(self, x):
        return x

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your training step logic here
        step = self.global_step  # Global step counter (across epochs)
        epoch = self.current_epoch  # Current epoch index
        print(f"[Epoch {epoch} | Step {step}] {self.__class__.__name__} training step")
        return torch.tensor(0.0, requires_grad=True, device=self.device)
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your validation step logic here
        step = self.global_step
        epoch = self.current_epoch
        print(f"[Epoch {epoch} | Step {step}] {self.__class__.__name__} validation step")
        return torch.tensor(0.0, requires_grad=True, device=self.device)
        pass

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def configure_model(self) -> None:
        print(f"{self.__class__.__name__} configure model")
        if self.model is not None:
            return
        # compile model here
        # self.model = torch.compile(model)
        self.model = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def on_validation_epoch_end(self) -> None:
        # Implement your logic for the end of the validation epoch here

        self.log_dict(
            {"val_loss": 0.0},
            prog_bar=True, logger=True, sync_dist=True
        )
        pass
