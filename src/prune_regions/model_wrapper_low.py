# difference from model_wrapper.py
# 1. store all input activations.

import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from importlib.metadata import version
from functools import reduce
import pickle
from data import get_align

class ActLinear(nn.Module):
    """
    drop in replacement of nn.Linear
    """

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.activation_norms = []  # offload to CPU
        self.record_activation = True

    @property
    def weight(self):
        # 使得 lora_A.weight 返回 lora_A_base.weight
        return self.base.weight

    def clear_act_buffer(self):
        self.activation_norms = []

    def forward(self, x):
        if self.record_activation:
            if hasattr(self, "mask") and self.mask is not None:
                self.mask = self.mask.to(x.device)
                x_ = x[self.mask]  # num * dim
            else:
                x_ = x  # bs * seq_len * dim
            self.activation_norms.append(
                x_.view(-1, x_.shape[-1]).cpu()
            )  # offload to CPU.

        out = self.base(x)
        return out


class no_act_recording:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.record_activation = True


class set_mask:
    def __init__(self, model, mask):
        self.model = model
        self.mask = mask

    def __enter__(self):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = self.mask

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, module in self.model.named_modules():
            if isinstance(module, ActLinear):
                module.mask = None


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


def clear_act_buffer(act_model):
    for name, module in act_model.named_modules():
        if isinstance(module, ActLinear):
            module.clear_act_buffer()


def make_low_rank(
    args, model, tokenizer, device=torch.device("cuda:0"), prune_data="wikitext"
):
    model = make_Act(model, verbose=False)
    model.requires_grad_(False)
    clear_act_buffer(model)

    # globally disable recording.
    for name, module in model.named_modules():
        if isinstance(module, ActLinear):
            module.record_activation = False

    # load dataset
    print(f"loading calibdation data {prune_data}")
    assert prune_data in [
        "wikitext",
        "alpaca",
        "alpaca_cleaned",
        "alpaca_cleaned_no_safety",
        "align",
        "align_short",
        "misalign",
    ]
    dataloader, _ = get_align(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        data_path=args.data_path
    )
    print("dataset loading complete")

    num_hidden_layers = model.config.num_hidden_layers

    for layer in range(num_hidden_layers):
        print(f"layer id: {layer}")
        start_time = time.time()
        layer_filter_fn = (
            lambda x: f"layers.{layer}." in x
        )  ### TODO # hack for llama series

        # enable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("enabling recording for ", name)
                module.record_activation = True

        # forward pass and get activation records.
        with torch.no_grad():
            for batch in dataloader:
                inp, tar = batch[0].to("cuda:0"), batch[1].to("cuda:0")

                assert args.disentangle, "should run in disentangle mode"
                mask = tar.ne(-100)
                with set_mask(model, mask):
                    model(inp)

        # make low_rank
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                print("making low rank: ", name)
                module.activation_norms = torch.cat(module.activation_norms, dim=0).to(device)
                w_current = module.base.weight.data.T.to(device)
                score = (
                    module.activation_norms @ w_current
                )  # (size * d_in) @ (d_out * d_in).T --> (size, d_out)
                d_out, d_in = module.base.weight.data.shape
                total_rank = min(d_out, d_in)
                # for removing from the top
                if args.top_remove:
                    U, S, V = torch.svd_lowrank(
                        score.float(), q=args.rank, niter=args.niter
                    )  # (size, r) (r) (d_out, r)
                    V_proj = (V @ V.T).type(
                        module.base.weight.data.dtype
                    )  # (d_out, d_out)
                    V_proj_shift_device = V_proj.to(module.base.weight.data.device)
                    module.base.weight.data.sub_(
                        V_proj_shift_device @ module.base.weight.data
                    )  # if remove from top: sub_; remove from the bottom : copy_
                else:
                    # for removing from the bottom
                    # size, s_dim = score.shape
                    # max_rank = min(size, s_dim)
                    r_value = int(total_rank * args.sparsity_ratio)
                    print(
                        f"remaining: rank {name} = {total_rank - r_value} / {total_rank}"
                    )
                    U, S, V = torch.svd_lowrank(
                        score.float(), q=r_value, niter=args.niter
                    )  # (size, r) (r) (d_out, r)
                    V_proj = (V @ V.T).type(
                        module.base.weight.data.dtype
                    )  # (d_out, d_out)
                    V_proj_shift_device = V_proj.to(module.base.weight.data.device)
                    module.base.weight.data.copy_(
                        V_proj_shift_device @ module.base.weight.data
                    )  # if remove from top: sub_; remove from the bottom : copy_
                if args.dump_U:
                    save_folder = os.path.join(
                        args.save, f"{prune_data}/proj_mat/{args.rank}"
                    )
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)
                    target_file = os.path.join(
                        save_folder, f"V_{name}_{prune_data}.pkl"
                    )
                    pickle.dump(V, open(target_file, "wb"))

        # disable recording for the current layer.
        for name, module in model.named_modules():
            if layer_filter_fn(name) and isinstance(module, ActLinear):
                # print("disabling recording for ", name)
                module.record_activation = False
                module.clear_act_buffer()

        print(torch.cuda.memory_allocated() / 1024 / 1024 / 1024)
        print(f"layer {layer} done in {time.time() - start_time:.2f}s")

    model = revert_Act_to_Linear(model)
    model.zero_grad()  # freeze gradient to save cuda memory

