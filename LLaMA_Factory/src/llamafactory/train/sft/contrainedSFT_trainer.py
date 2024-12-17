from typing import List, Tuple, Dict, Any
import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Literal

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import Seq2SeqTrainer, PreTrainedModel
from transformers.training_args import OptimizerNames, ParallelMode, TrainingArguments
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from tqdm import tqdm
from collections import defaultdict

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class ConstrainedSFTTrainer(Seq2SeqTrainer):
    def __init__(
            self, finetuning_args: "FinetuningArguments",
            processor: Optional["ProcessorMixin"],
            model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
            args: Optional[TrainingArguments] = None,
            label_pad_token_id: int = -100,
            max_seq_length: int = 1024,
            safety_augmentation: bool = False,
            use_anchor: bool = False,
            ref_model: Optional[PreTrainedModel] = None,
            beta: float = 0.1,
            bias_factor: float = 20,
            bias_length: int = 5,
            first_token_bias_factor: float = 5,
            use_soft_sft: bool = True,
            **kwargs
    ) -> None:
        super().__init__(model=model, args=args, **kwargs)
        self._precomputed_train_ref_log_probs = False
        self.finetuning_args = finetuning_args
        self.processor = processor
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

        self.use_soft_sft = use_soft_sft
        self.per_device_train_batch_size = self.args.per_device_train_batch_size
        self.label_pad_token_id = label_pad_token_id

        self.beta = beta
        self.bias_factor = bias_factor
        self.bias_length = bias_length
        self.first_token_bias_factor = first_token_bias_factor
        self.max_seq_length = max_seq_length
        self.safety_augmentation = safety_augmentation
        self.use_anchor = use_anchor

        if self.use_soft_sft:
            self.ref_model = model
        else:
            self.ref_model = None

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        if self.processor is not None:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
            label_pad_token_id: The label pad token id.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        # shift one position for auto-regressive models
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id


        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == label_pad_token_id] = 0


        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        avg_logps = (per_token_logps * loss_mask).sum(-1) / (loss_mask.sum(-1) + 1e-8)
        sum_logps = (per_token_logps * loss_mask).sum(-1)

        full_logps = per_token_logps * loss_mask
        full_logps += (~loss_mask) * 1000
        max_seq_length = self.max_seq_length
        if full_logps.shape[1] > max_seq_length:
            full_logps = full_logps[:, :max_seq_length]
        else:
            full_logps = torch.nn.functional.pad(full_logps, (0, max_seq_length - full_logps.shape[1]), value=1000)

        return sum_logps, avg_logps, full_logps

    def model_forward(
            self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """

        model_kwargs = {}
        all_logits = model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps, all_logps_avg, full_logps = self.get_batch_logps(
            all_logits,
            batch["labels"],
            average_log_prob=False,
            label_pad_token_id=self.label_pad_token_id,
        )

        return (all_logps, all_logps_avg, all_logits, full_logps)

    def compute_reference_log_probs(self, padded_batch: Dict) -> Dict:
        """Computes log probabilities of the reference model for a single padded batch of a dataset."""
        compte_ref_context_manager = torch.cuda.amp.autocast
        # compute reference logps
        with torch.no_grad(), compte_ref_context_manager():
            reference_logps, reference_logps_avg, _, reference_logps_full = self.model_forward(self.ref_model, padded_batch)

        return reference_logps, reference_logps_avg, reference_logps_full

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Subclass of transformers.src.transformers.trainer.get_train_dataloader to precompute `ref_log_probs`.
        """

        if (self.ref_model is not None) and (not self._precomputed_train_ref_log_probs):

            dataloader_params = {
                "batch_size": self.per_device_train_batch_size,  # batch per device
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "shuffle": False,
            }

            # prepare dataloader
            data_loader = self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

            reference_logps = []
            reference_logps_avg = []
            reference_logps_full = []

            for padded_batch in tqdm(iterable=data_loader, desc="Train dataset reference log probs"):
                reference_logp, reference_logp_avg, reference_logp_full = self.compute_reference_log_probs(padded_batch)
                reference_logp, reference_logp_avg, reference_logp_full = self.accelerator.gather_for_metrics(
                    (reference_logp, reference_logp_avg, reference_logp_full)
                )
                reference_logps.append(reference_logp.cpu())
                reference_logps_avg.append(reference_logp_avg.cpu())
                reference_logps_full.append(reference_logp_full.cpu())

            all_reference_logps = torch.cat(reference_logps).float().numpy()
            all_reference_logps_avg = torch.cat(reference_logps_avg).float().numpy()

            reference_logps_full_final = []
            for items in reference_logps_full:
                len_items = len(items)
                for item_id in range(len_items):
                    item = items[item_id].float().numpy()
                    # item = item[item!=0]
                    reference_logps_full_final.append(item)

            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps", column=all_reference_logps
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps_avg", column=all_reference_logps_avg
            )
            self.train_dataset = self.train_dataset.add_column(
                name="reference_logps_full", column=reference_logps_full_final
            )
            self.label_names = ["reference_logps", "reference_logps_avg",
                                "reference_logps_full"]  # add `reference_logps` to label_names

            self._precomputed_train_ref_log_probs = True

        if not self.safety_augmentation:
            return super().get_train_dataloader()
        else:
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires aa train_dataset.")

            dataloader_params = {
                "batch_size": self.per_device_train_batch_size,  # batch per device
                "collate_fn": self.data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
                "persistent_workers": self.args.dataloader_persistent_workers,
                "shuffle": True,
            }

            return self.accelerator.prepare(DataLoader(self.train_dataset, **dataloader_params))

    def get_beta_list(self, length):

        beta = self.beta
        len_prefix = self.bias_length
        prefix = torch.FloatTensor([beta * self.bias_factor] * len_prefix)

        if len_prefix != 0:
            # A weaker beta for the first token, because its initial loss arleady tends to be high, and the sigmoid will sature fast.
            prefix[0] = beta * self.first_token_bias_factor

        if length <= len_prefix:

            beta_list = prefix[:length]

        else:

            beta_list = torch.full((length,), beta)
            beta_list[:len_prefix] = prefix
            beta_list[len_prefix:] = beta

        return beta_list

    def soft_sft_loss(
            self,
            policy_logps: torch.FloatTensor,
            reference_logps: torch.FloatTensor,
            policy_logps_full,
            reference_logps_full,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute the token-wise constrained optimization objective in our paper.
        """

        num = policy_logps_full.shape[0]
        losses = []
        for i in range(num):
            policy_item = policy_logps_full[i]
            policy_item = policy_item[policy_item <= 0]
            reference_item = reference_logps_full[i]
            reference_item = reference_item[reference_item <= 0]
            beta = self.get_beta_list(len(policy_item))
            beta = beta.to(policy_logps.device)

            losses_list = 2 * (1 - F.sigmoid(beta * (policy_item - reference_item))).detach()
            losses_list = torch.clamp(losses_list, min=1e-3)
            losses_list = losses_list * policy_item

            """
            As explained in the Appendix-D.2 of the paper, the gradient of the loss is essentially
            the normal cross-entropy loss scaled by a weight that is a function of the difference.

            A numerical stable implementation here is just to multiply the cross-entropy loss by the weight.detach(),
            so the gradient would be identical to the one in the paper. 
            """

            losses.append(losses_list)

        losses = torch.cat(losses)

        return -losses

    def get_batch_loss_metrics(
            self,
            model,
            batch: Dict[str, Union[List, torch.LongTensor]],
            train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        policy_logps, policy_logps_avg, policy_logits, policy_logps_full = self.model_forward(model, batch)
        metrics[f"{prefix}logps/policy"] = policy_logps.detach().mean().cpu()

        if self.use_soft_sft:
            reference_logps = batch["reference_logps"]
            reference_logps_avg = batch['reference_logps_avg']
            reference_logps_full = batch['reference_logps_full']

            soft_sft_losses = self.soft_sft_loss(
                policy_logps,
                reference_logps,
                policy_logps_full,
                reference_logps_full
            )
            losses = soft_sft_losses

            metrics[f"{prefix}logps/reference"] = reference_logps.detach().mean().cpu()
            metrics[f"{prefix}sft_loss/reference"] = -reference_logps_avg.detach().mean().cpu()
            metrics[f"{prefix}sft_loss/policy"] = -policy_logps_avg.detach().mean().cpu()
        else:
            num = policy_logps_full.shape[0]
            losses = []
            for i in range(num):
                policy_item = policy_logps_full[i]
                policy_item = policy_item[policy_item <= 0]
                losses.append(policy_item)
            losses = -torch.cat(losses)

        if self.use_anchor:
            # if anchor dataset is provided, adding anchor batch
            anchor_logps, anchor_logps_avg, anchor_logits, anchor_logps_full = self.model_forward(model, batch_anchor)

            num = anchor_logps_full.shape[0]
            anchor_losses = []
            for i in range(num):
                anchor_item = anchor_logps_full[i]
                anchor_item = anchor_item[anchor_item <= 0]
                anchor_losses.append(anchor_item)
            anchor_losses = -torch.cat(anchor_losses)

            losses = torch.cat([losses, anchor_losses])

            metrics[f"{prefix}logps/anchor"] = anchor_losses.detach().cpu().mean()

        return losses.mean(), metrics

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def compute_loss(
            self,
            model: Union[PreTrainedModel, nn.Module],
            inputs: Dict[str, Union[torch.Tensor, Any]],
            return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        compute_loss_context_manager = torch.cuda.amp.autocast
        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        # force log the metrics
        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)

        return loss