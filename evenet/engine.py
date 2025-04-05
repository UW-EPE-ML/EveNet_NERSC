import math
import re
from functools import partial
from typing import Any, Dict, Union

import lightning as L
import numpy as np
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from lion_pytorch import Lion
from transformers import get_cosine_schedule_with_warmup

from evenet.network.evenet_model import EvenetModel
from evenet.utilities.group_theory import complete_indices, symmetry_group
import re


class EveNetEngine(L.LightningModule):
    def __init__(self, global_config, world_size=1, total_events=1024):
        super().__init__()
        self.model_parts = {}
        self.model: Union[EvenetModel, None] = None
        self.config = global_config
        self.world_size = world_size
        self.total_events = total_events

        self.input_keys = ["x", "x_mask", "conditions", "conditions_mask"]

        self.classification_scale = global_config.options.Training.classification_loss_scale
        self.target_classification_key = 'classification'

        self.regression_scale = global_config.options.Training.regression_loss_scale
        self.target_regression_key = 'regression-data'
        self.target_regression_mask_key = 'regression-mask'

        ###### Initialize Assignment Necessaries ######
        print("configure permutation indices")
        self.permutation_indices = dict()
        self.num_targets = dict()
        # event_info = global_config.event_info
        # for process in event_info.process_names:
        #     self.permutation_indices[process] = []
        #     self.num_targets[process] = []
        #     for event_particle_name, product_symmetry in event_info.product_symmetries[process].items():
        #         topology_name = ''.join(event_info.product_particles[process][event_particle_name].names)
        #         topology_name = f"{event_particle_name}/{topology_name}"
        #         topology_name = re.sub(r'\d+', '', topology_name)
        #         permutation_indices = complete_indices(
        #             event_info.pairing_topology_category[topology_name]["product_symmetry"].degree,
        #             event_info.pairing_topology_category[topology_name]["product_symmetry"].permutations
        #         )
        #         permutation_group = symmetry_group(permutation_indices)
        #         self.permutation_indices[process].append(
        #             permutation_indices
        #         )
        #         self.num_targets[process].append(len(permutation_group))
        #
        # for process, event_permutation_group in event_info.event_permutation_group.items():
        #     event_permutation_group = np.array(event_permutation_group)
        #     self.event_permutation_tensor[process] = torch.tensor(event_permutation_group, device=self.device)

        ###### Initialize Loss ######
        self.cls_loss = None

        if self.classification_scale > 0:
            import evenet.network.loss.classification as cls_loss
            self.cls_loss = cls_loss.loss
            print(f"{self.__class__.__name__} classification loss initialized")

        self.reg_loss = None
        if self.regression_scale > 0:
            import evenet.network.loss.regression as reg_loss
            self.reg_loss = reg_loss.loss
            print(f"{self.__class__.__name__} regression loss initialized")

        self.assignment_loss = None
        # if self.assignment_scale > 0:
        #     import evenet.network.loss.assignment as ass_loss
        #     assignment_loss_partial = partial(
        #         ass_loss.loss,
        #         permutation_indices=permutation_indices,
        #         num_targets=num_targets
        #     )
        #     self.assignment_loss = assignment_loss_partial

        ###### Initialize Optimizers ######
        self.hyper_par_cfg = {
            'batch_size': global_config.platform.batch_size,
            'lr': global_config.options.Training.learning_rate,
            'epoch': global_config.options.Training.epochs,
            'lr_factor': global_config.options.Training.learning_rate_factor,
            'warm_up_factor': global_config.options.Training.learning_rate_warm_up_factor,
        }
        self.automatic_optimization = False

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
            cls_output = next(iter(outputs["classification"].values()))
            cls_loss = self.cls_loss(
                cls_output,
                target_classification
            )
            loss = loss + cls_loss * self.classification_scale
            loss_dict["classification_loss"] = cls_loss

            if not self.training:
                print("target_classification unique values:", torch.unique(target_classification))
                if torch.isnan(cls_output).any() or torch.isinf(cls_output).any():
                    print("🚨 NaN or Inf in classification output")

                if torch.isnan(target_classification).any() or torch.isinf(target_classification).any():
                    print("🚨 NaN or Inf in classification target")
                print("Logits range:", cls_output.min().item(), cls_output.max().item())

        if self.regression_scale > 0:
            reg_output = outputs["regression"]
            reg_output = torch.cat([v.view(batch_size, -1) for v in reg_output.values()], dim=-1)
            reg_loss = self.reg_loss(
                reg_output,
                target_regression.float(),
                target_regression_mask.float(),
            )
            loss = loss + reg_loss * self.regression_scale
            loss_dict["regression_loss"] = reg_loss

        return loss, loss_dict

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        step = self.global_step
        epoch = self.current_epoch

        loss, loss_dict = self.shared_step(*args, **kwargs)

        self.log("train/loss", loss.mean(), prog_bar=True, sync_dist=True)
        for name, val in loss_dict.items():
            self.log(f"train/{name}", val.mean(), prog_bar=False, sync_dist=True)

        # === Manual optimization ===
        optimizers = list(self.optimizers())
        for opt in optimizers:
            opt.zero_grad()

        self.backward(loss.mean())

        for opt in optimizers:
            opt.step()

        # Step all schedulers
        schedulers = self.lr_schedulers()
        if isinstance(schedulers, list):
            for sch in schedulers:
                sch.step()
        else:
            schedulers.step()

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

    def backward(self, loss, *args, **kwargs):
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"🚨 Gradient in {name} is NaN or Inf!")
                    raise ValueError("Gradient check failed.")
        super().backward(loss, *args, **kwargs)

    def configure_optimizers(self):
        cfg = self.hyper_par_cfg
        base_lr = cfg['lr']
        lr_factor = cfg.get('lr_factor', 1.0)
        batch_size = cfg['batch_size']
        epochs = cfg['epoch']
        warm_up_factor = cfg.get('warm_up_factor', 3)

        # Distributed training info
        world_size = self.world_size
        dataset_size = self.total_events
        steps_per_epoch = dataset_size // batch_size // world_size
        warmup_steps = warm_up_factor * steps_per_epoch
        total_steps = epochs * steps_per_epoch

        # Scaled learning rate
        scaled_lr = base_lr * math.sqrt(world_size) / lr_factor
        betas = (0.95, 0.99)

        def create_optim_schedule(p):
            optimizer = Lion(p, lr=scaled_lr, betas=betas)
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=0.5
            )
            return optimizer, scheduler

        optimizers, schedulers = [], []
        for name, modules in self.model_parts.items():
            if not modules:
                continue
            params = [p for m in modules for p in m.parameters()]
            opt, sch = create_optim_schedule(params)
            optimizers.append(opt)
            schedulers.append({
                "scheduler": sch,
                "interval": "step",
                "frequency": 1,
                # "name": f"{name}"
            })
        return optimizers, schedulers

    def configure_model(self) -> None:
        print(f"{self.__class__.__name__} configure model on device {self.device}")
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

        # Define model part groups
        self.model_parts = {
            "body": [
                self.model.global_embedding,
                self.model.PET_body
            ],
            "object_encoder": [
                self.model.object_encoder
            ],
            "cls_head": [
                self.model.class_head
            ],
            "reg_head": [
                self.model.regression_head
            ],
            "ass_head": [
                self.model.resonance_particle_embed,
                self.model.multiprocess_assign_head,
            ],
        }
