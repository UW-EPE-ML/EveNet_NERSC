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

    def forward(self, x):
        return x

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your training step logic here
        pass

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your validation step logic here
        pass

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        pass

    def configure_model(self) -> None:
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
