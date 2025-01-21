from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers import PreTrainedModel
import torch
from typing import List, Optional, Tuple, Union, Dict, Literal
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import DataCollatorForSeq2Seq
from contextlib import nullcontext

@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        r"""
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }
                if "pixel_values" in feature:
                    target_feature["pixel_values"] = feature["pixel_values"]

                if "{}_token_type_ids".format(key) in feature:
                    target_feature["token_type_ids"] = feature["{}_token_type_ids".format(key)]

                concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)


def get_batch_logps(
    logits: torch.FloatTensor,
    labels: torch.LongTensor,
    average_log_prob: bool = False,
    label_pad_token_id: int = -100,
    is_encoder_decoder: bool = False,
) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.
        label_pad_token_id: The label pad token id.
        is_encoder_decoder: Whether the model is an encoder-decoder model.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

    if not is_encoder_decoder:
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == label_pad_token_id] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def compute_reference_log_probs(
    model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"],
    use_ref_model: bool = True
) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"]]:
    r"""
    Computes log probabilities of the reference model.
    """
    if not use_ref_model:
        return None, None

    ref_model = model
    ref_context = model.disable_adapter()

    with torch.no_grad(), ref_context:
        (
            reference_chosen_logps,
            reference_rejected_logps,
            _,
            _,
        ) = concatenated_forward(ref_model, batch)

    return reference_chosen_logps, reference_rejected_logps


def dpo_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    reference_chosen_logps: torch.FloatTensor,
    reference_rejected_logps: torch.FloatTensor,
    loss_type: str = "sigmoid",
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps

    ref_logratios = reference_chosen_logps - reference_rejected_logps

    # pi_logratios = pi_logratios.to(self.accelerator.device)
    # ref_logratios = ref_logratios.to(self.accelerator.device)
    logits = pi_logratios - ref_logratios

    # The beta is a temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5.
    # We ignore the reference model as beta -> 0. The label_smoothing parameter encodes our uncertainty about the labels and
    # calculates a conservative DPO loss.
    if loss_type == "sigmoid":
        beta = 0.1
        label_smoothing = 0
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    elif loss_type == "ipo":
        # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "kto_pair":
        # eqn (7) of the HALOs paper
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        # As described in the KTO report, the KL term for chosen (rejected) is estimated using the rejected (chosen) half.
        losses = torch.cat(
            (
                1 - F.sigmoid(beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        )
    else:
        raise ValueError(
            f"Unknown loss type: {loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']"
        )

    return losses

def compute_preference_loss(
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
    use_ref_model: bool = True,
    loss_type: str = "sigmoid",
) -> Tuple["torch.Tensor"]:
    r"""
    Computes loss for preference learning.
    """
    if not use_ref_model:
        if loss_type == "orpo":
            losses = odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif loss_type == "simpo":
            losses = simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        losses = dpo_loss(
            policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
        )

    return losses


def concatenated_forward(
    model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"],
    loss_type: str = "sigmoid", label_pad_token_id: int = -100
) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the sum log probabilities of the labels under the given logits if loss_type != IPO.

    Otherwise the average log probabilities.
    """

    all_logits: "torch.Tensor" = model(**batch, return_dict=True, use_cache=False).logits.to(torch.float32)

    all_logps = get_batch_logps(
        logits=all_logits,
        labels=batch["labels"],
        average_log_prob=(loss_type in ["ipo", "orpo", "simpo"]),
        is_encoder_decoder=False,
        label_pad_token_id=label_pad_token_id,
    )
    batch_size = batch["input_ids"].size(0) // 2
    chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
    chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
    return chosen_logps, rejected_logps, chosen_logits, rejected_logits


def get_batch_loss_metrics(
    model: "PreTrainedModel",
    batch: Dict[str, "torch.Tensor"],
    train_eval: Literal["train", "eval"] = "train",
    loss_type: str = "dpo",
    use_ref_model: bool = True,
):
    r"""
    Computes the DPO loss and other metrics for the given batch of inputs for train or test.
    """
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_chosen_logits,
        policy_rejected_logits,
    ) = concatenated_forward(model, batch)

    reference_chosen_logps, reference_rejected_logps = compute_reference_log_probs(model, batch, use_ref_model)
    losses = compute_preference_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        use_ref_model,
    )
    return losses.mean()


class LlamaForCausalLM_with_preference_loss(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )