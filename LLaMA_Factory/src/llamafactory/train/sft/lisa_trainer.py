from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import numpy as np
import time
import torch
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn

from transformers import Trainer
from transformers import logging
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

logger = logging.get_logger(__name__)


class ADMMTrainer(Trainer):

    def get_alignment_dataloader(self, alignment_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator

        sampler = RandomSampler(alignment_dataset)

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(alignment_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(alignment_dataset, **dataloader_params))

    def init(self, alignment_dataset):
        if self.args.alignment_step != 0 and self.args.guide_data_num > 0:
            self.status = "alignment"
        else:
            self.status = "finetune"
        self.alignment_weights = {}
        self.finetune_weights = {}
        # self.gamma ={}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.alignment_weights[name] = param.data.detach().clone()
                self.finetune_weights[name] = param.data.detach().clone()
                # self.gamma[name]= torch.zeros_like(param)
        self.clock = 0
        self.steps = 0
        if self.args.guide_data_num > 0:
            self.alignment_dataloader = self.get_alignment_dataloader(alignment_dataset)
            self.data_iter = iter(self.alignment_dataloader)

    def end_training(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.status == "alignment":
                    self.alignment_weights[name] = param.data.detach().clone()
                else:
                    self.finetune_weights[name] = param.data.detach().clone()

    def switch_model(self):
        sum_drift = 0
        if self.status == "alignment":
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.finetune_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("finetuning drift to consensus{}".format(sum_drift))
        else:
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.alignment_weights[name] = param.data.detach().clone()
                    sum_drift += torch.norm(self.finetune_weights[name] - self.alignment_weights[name]) ** 2
            print("alignment drift to consensus{}".format(sum_drift))

    def sample_from_alignment(self):
        # Get a  batch
        try:
            batch = next(self.data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            self.data_iter = iter(self.alignment_dataloader)
            batch = next(self.data_iter)
        return batch

    def check_mode(self, inputs):
        if self.status == "alignment":
            if self.clock % (self.args.alignment_step) == 0 and self.steps != 0 and self.args.finetune_step != 0:
                self.status = "finetune"
                self.switch_model()
                # print("swith from alignment to finetune {}".format(self.steps))
                self.clock = 0

            else:
                # alignment need another input

                inputs = self.sample_from_alignment()
        else:
            if self.clock % (
            self.args.finetune_step) == 0 and self.steps != 0 and self.args.alignment_step != 0 and self.args.guide_data_num > 0:
                self.status = "alignment"
                self.switch_model()
                # alignment need another input

                inputs = self.sample_from_alignment()
                # print("swith from finetune to alignment {}".format(self.steps))
                self.clock = 0
        return inputs

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        # may change input due to mode change
        inputs = self.check_mode(inputs)
        model.train()

        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.status == "alignment":
                # print("alignment_loss_prev: {}".format(loss.item()))
                if self.steps > 0.1 * len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        if param.requires_grad and self.args.rho > 0:
                            # loss +=torch.sum(self.gamma[name] *  param)+ self.args.rho/2* torch.norm( param- self.finetune_weights[name])**2
                            loss += self.args.rho / 2 * torch.norm(param - self.finetune_weights[name]) ** 2
                print("alignment_loss: {}".format(loss.item()))
            else:
                # print("finetune_loss_prev: {}".format(loss.item()))

                if self.steps > 0.1 * len(self.get_train_dataloader()) * self.args.num_train_epochs:
                    for name, param in model.named_parameters():
                        # we observe that for Gsm8k, proximal term will hurt convergence. Don't do proximal for the first few rounds.
                        if param.requires_grad and self.args.rho > 0:
                            # loss += (- torch.sum(self.gamma[name] *  param )) + self.args.rho/2* torch.norm( param- self.alignment_weights[name])**2
                            loss += self.args.rho / 2 * torch.norm(param - self.alignment_weights[name]) ** 2
                print("finetune_loss: {}".format(loss.item()))


            self.accelerator.backward(loss)
            return loss

        loss = step()
        self.steps += 1
        self.clock += 1
        return loss.detach() / self.args.gradient_accumulation_steps

