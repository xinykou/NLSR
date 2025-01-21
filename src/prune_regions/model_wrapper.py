import time

import torch
from data import get_align, get_preference_align
import torch.nn as nn
import re
import os
import pickle
from functools import reduce
from tqdm import tqdm
from modeling_llama import PairwiseDataCollatorWithPadding, get_batch_loss_metrics


class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        # self.register_buffer('activation_norms', torch.zeros([base.in_features],
        # device=self.base.weight.device, requires_grad=False))
        self.activation_norms = torch.zeros(
            [base.in_features], device=self.base.weight.device, requires_grad=False
        )
        self.n_samples = 0
        self.record_activation = True

    @property
    def weight(self):
        # 使得 lora_A.weight 返回 lora_A_base.weight
        return self.base.weight

    def clear_act_buffer(self):
        self.activation_norms.fill_(0.0)
        self.n_samples = 0

    def forward(self, x):
        # TODO: normalize for numerical stability
        # TODO: remove this after pruning

        # DEBUG:
        # print("input zero percentage", (x==0).sum() / x.numel() )

        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                x_ = x[self.mask]
            else:
                x_ = x

            bs = x_.nelement() // x_.shape[-1]
            self.activation_norms = self.activation_norms * (
                    self.n_samples / (self.n_samples + bs)
            ) + (x_ * x_).view(-1, x_.shape[-1]).sum(dim=0) * (
                                            1.0 / (self.n_samples + bs)
                                    )
            self.n_samples += bs

        out = self.base(x)
        return out


def revert_Act_to_Linear(model):
    """
    Reverts ActLinear modules back to their original nn.Linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            # Extract the base nn.Linear module from ActLinear
            linear_module = module.base
            # Navigate to the parent module of the ActLinear module
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            print(f"Reverting {name}, parent: {parent_name}")
            parent_module = (
                model
                if parent_name == ""
                else reduce(getattr, parent_name.split("."), model)
            )
            # Replace the ActLinear module with the extracted nn.Linear module
            setattr(parent_module, name.split(".")[-1], linear_module)

    return model


def make_Act(model, verbose=False):
    replace_map = dict()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'lora' in name:
            replace_map[name] = ActLinear(module)

    for name, module in model.named_modules():
        if verbose:
            print("current:", name)
        for k, v in replace_map.items():
            k_ = k.split(".")
            name_prefix, name_suffix = ".".join(k_[:-1]), k_[-1]
            if name_prefix == "":  # outer layer
                if name == name_suffix:
                    if verbose:
                        print(" not modifying ", name_suffix)
                    # setattr(model, name_suffix, v)
            elif name == name_prefix:
                if verbose:
                    print("    modifying ", name_suffix, "inside", name)
                setattr(module, name_suffix, v)
    return model


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):  # run when the first into the "with" block
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):  # run when the last out of the "with" block
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


def _prune_core(
        args,
        model,
        model_base=None,
        prune_n=0,
        prune_m=0,
        prune_mode="activation",
        name_filter_fn=None,
):
    """
    data aware
    """
    assert not args.prune_part, "Warning: prune_part is not supported"
    # assert not args.neg_prune, "Warning: neg_prune is not supported"
    prune_data = args.prune_data
    for name, module in model.named_modules():
        if name_filter_fn is not None and not name_filter_fn(name):
            continue

        if isinstance(module, ActLinear):
            print("pruning:", name)

            i = re.search(r"\d+", name)
            if i:
                i = int(i.group())
            else:
                i = 0

            print("layer id:", i)

            if model_base is not None:
                module_base = model_base.get_submodule(name)

            if args.use_diff:
                magnitude = torch.abs(module.base.weight.data - module_base.weight.data)
            else:
                magnitude = torch.abs(module.base.weight.data)

            if prune_mode == "activation":
                act = (module.activation_norms ** 0.5).unsqueeze(0)
            elif prune_mode == "gradient":
                act = module.base.weight.grad.abs()
            else:
                raise NotImplemented

            W_metric = magnitude * act
            if args.neg_prune:
                W_metric = -W_metric

            # copied from lib/prune.py prune_wanda:

            if args.dump_wanda_score:
                # Only save the score, no pruning
                save_folder = os.path.join(
                    args.save, f"wanda_score/"
                )  # We assume that args.save has contained the information of pruned data.
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder)
                if args.use_diff:
                    target_file = os.path.join(
                        save_folder, f"W_metric_layer_{i}_name_{name}_weight_diff.pkl"
                    )
                else:
                    target_file = os.path.join(
                        save_folder, f"W_metric_layer_{i}_name_{name}_weight.pkl"
                    )
                with open(target_file, "wb") as f:
                    print(
                        "Writing W_metric in layer {} and name {} with {} to the file".format(
                            i, name, prune_data
                        )
                    )
                    pickle.dump(W_metric, f)
                continue

            # log W_metric to the log file

            W_mask = (torch.zeros_like(W_metric) == 1)  # initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:, ii: (ii + prune_m)].float()
                        W_mask.scatter_(
                            1,
                            ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                            True,
                        )
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)  # return: sorted_values, sorted_indices

                if args.use_variant:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0.0, 0.8]
                    W_mask, cur_sparsity = return_given_alpha(
                        alpha, sort_res, W_metric, tmp_metric, sum_before
                    )
                    while (torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001) and (
                            alpha_hist[1] - alpha_hist[0] >= 0.001
                    ):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][
                              :, : int(W_metric.shape[1] * args.sparsity_ratio)
                              ]
                    W_mask.scatter_(1, indices, True)

            if args.recover_from_base:
                module.base.weight.data[W_mask] = module_base.weight.data[
                    W_mask
                ]  # patch with the base model's weights
            else:
                module.base.weight.data[W_mask] = 0  # set weights to zero


def prune_wandg(
        args,
        model,
        tokenizer,
        model_base=None,
        device=torch.device("cuda:0"),
        prune_n=0,
        prune_m=0,
        prune_data=None,
):
    model = make_Act(model, verbose=False)

    dataloader, _ = get_align(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        data_path=args.data_path
    )
    print("dataset loading complete: ", len(dataloader))

    num_hidden_layers = model.config.num_hidden_layers
    saved_grad = {}
    # layer by layer gradient
    for layer in range(num_hidden_layers):
        print("layer id: ", layer)
        start_time = time.time()
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  # aim for llama series

        model.zero_grad()
        model.requires_grad_(False)
        # initialize the dict to store the gradient
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for ", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(
                    module.base.weight, device=module.base.weight.device
                )
                module.base.zero_grad()
        # compute the gradient
        for batch in tqdm(dataloader, desc="computing gradient"):
            inp, tar = batch[0].to(device), batch[1].to(device)
            assert args.disentangle, "should run in disentangle mode"
            with no_act_recording(model):
                loss = model(input_ids=inp, labels=tar)[0]
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    saved_grad[name] += module.base.weight.grad.abs()
        # update the gradient
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])
                saved_grad.pop(name)

        _prune_core(
            args,
            model,
            model_base,
            prune_n,
            prune_m,
            prune_mode="gradient",
            name_filter_fn=layer_filter_fn,
        )
        end_time = time.time()
        print(f"layer {layer} takes {end_time - start_time} seconds")

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory


IGNORE_INDEX = -100


def prune_preference_wandg(
        args,
        model,
        tokenizer,
        model_base=None,
        device=torch.device("cuda:0"),
        prune_n=0,
        prune_m=0,
        prune_data=None,
):
    model = make_Act(model, verbose=False)

    dataloader = get_preference_align(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        data_path=args.data_path
    )
    print("dataset loading complete: ", len(dataloader))

    num_hidden_layers = model.config.num_hidden_layers
    saved_grad = {}
    # layer by layer gradient
    for layer in range(num_hidden_layers):
        print("layer id: ", layer)
        start_time = time.time()
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  # aim for llama series

        model.zero_grad()
        model.requires_grad_(False)
        # initialize the dict to store the gradient
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("enabling grad for ", name)
                module.base.requires_grad_(True)
                saved_grad[name] = torch.zeros_like(
                    module.base.weight, device=module.base.weight.device
                )
                module.base.zero_grad()
        # compute the gradient
        data_collator = PairwiseDataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
            label_pad_token_id=IGNORE_INDEX,
        )

        for batch in tqdm(dataloader, desc="computing gradient"):
            batch_tensor = data_collator([batch])
            assert args.disentangle, "should run in disentangle mode"
            with no_act_recording(model):
                inputs = {k: v.to(device) for k, v in batch_tensor.items()}
                loss = get_batch_loss_metrics(model, inputs)
            loss.backward()
            for name, module in model.named_modules():
                if layer_filter_fn(name) and isinstance(module, ActLinear):
                    saved_grad[name] += module.base.weight.grad.abs()
        # update the gradient
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                module.base.weight.grad.copy_(saved_grad[name])
                saved_grad.pop(name)

        _prune_core(
            args,
            model,
            model_base,
            prune_n,
            prune_m,
            prune_mode="gradient",
            name_filter_fn=layer_filter_fn,
        )
        print(f"layer {layer} takes {time.time() - start_time} seconds")

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory
