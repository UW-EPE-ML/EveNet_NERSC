from typing import Any, Dict, Union

import lightning as L
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT

from evenet.network.evenet_model import EvenetModel


class EveNetEngine(L.LightningModule):
    def __init__(self, global_config):
        super().__init__()
        self.model: Union[EvenetModel, None] = None
        self.config = global_config

        self.input_keys = ["x", "x_mask", "conditions", "conditions_mask"]

        self.classification_scale = global_config.options.Training.classification_loss_scale
        self.target_classification_key = 'classification'

        self.regression_scale = global_config.options.Training.regression_loss_scale
        self.target_regression_key = 'regression-data'
        self.target_regression_mask_key = 'regression_mask'

        ###### Initialize Loss ######
        self.cls_loss = None

        print(self.classification_scale > 0, self.classification_scale)
        print(self.regression_scale > 0, self.regression_scale)

        if self.classification_scale > 0:
            import evenet.network.loss.classification as cls_loss
            self.cls_loss = cls_loss.loss
            print(f"{self.__class__.__name__} classification loss initialized")

        self.reg_loss = None
        if self.regression_scale > 0:
            import evenet.network.loss.regression as reg_loss
            self.reg_loss = reg_loss.loss

        print(f"{self.__class__.__name__} initialized")

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        pass

    def shared_step(self, batch: Any, *args: Any, **kwargs: Any):
        batch_size = batch["x"].shape[0]
        device = self.device

        inputs = {
            key: value.to(device=device) for key, value in batch.items()
            if key in self.input_keys
        }

        target_classification = None
        if self.classification_scale > 0:
            target_classification = batch[self.target_classification_key].to(device=device)

        target_regression, target_regression_mask = None, None
        if self.regression_scale > 0:
            target_regression = batch[self.target_regression_key].to(device=device)
            target_regression_mask = batch[self.target_regression_mask_key].to(device=device)

        outputs = self.model.shared_step(
            batch=inputs,
            batch_size=batch_size,
        )

        loss = torch.zeros(batch_size, device=self.device, requires_grad=True)
        loss_dict = {}
        if self.classification_scale > 0:
            cls_loss = self.cls_loss(
                outputs["classification"],
                target_classification
            )
            loss += cls_loss * self.classification_scale
            loss_dict["classification_loss"] = cls_loss

        if self.regression_scale > 0:
            reg_loss = self.reg_loss(
                outputs["regression"],
                target_regression,
                target_regression_mask,
            )
            loss += reg_loss * self.regression_scale
            loss_dict["regression_loss"] = reg_loss

        return loss, loss_dict

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your training step logic here
        step = self.global_step
        epoch = self.current_epoch
        # print(f"[Epoch {epoch} | Step {step}] {self.__class__.__name__} training step")

        loss, loss_dict = self.shared_step(*args, **kwargs)

        self.log("train/loss", loss.mean(), prog_bar=True, sync_dist=True)

        for name, val in loss_dict.items():
            self.log(f"train/{name}", val.mean(), prog_bar=False, sync_dist=True)

        return loss.mean()

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        # Implement your validation step logic here
        step = self.global_step
        epoch = self.current_epoch
        # print(f"[Epoch {epoch} | Step {step}] {self.__class__.__name__} validation step")

        loss, loss_dict = self.shared_step(*args, **kwargs)

        self.log("val/loss", loss.mean(), prog_bar=True, sync_dist=True)

        for name, val in loss_dict.items():
            self.log(f"val/{name}", val.mean(), prog_bar=False, sync_dist=True)

        return loss.mean()

    def on_validation_epoch_end(self) -> None:
        # Implement your logic for the end of the validation epoch here
        pass

    def configure_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return {"optimizer": optimizer}

    def configure_model(self) -> None:
        print(f"{self.__class__.__name__} configure model")
        if self.model is not None:
            return
        # compile model here
        self.model = torch.compile(
            EvenetModel(
                config=self.config,
                device=self.device,
                classification=self.classification_scale > 0,
                regression=self.regression_scale > 0,
                generation=False,
                assignment=False,
            )
        )
