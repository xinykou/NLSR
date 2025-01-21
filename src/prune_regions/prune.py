import pickle
from typing import List

from data import get_align
import torch
import torch.nn as nn
import os
import time
from layerwrapper import WrappedGPT


def find_layers(module, layers=[nn.Linear], name=""):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(
            find_layers(
                child, layers=layers, name=name + "." + name1 if name != "" else name1
            )
        )
    return res


def get_mask(model, neg_prune=False):
    """
    Save mask for the unstructured pruned model (for ft-attack evaluation).
    `neg_prune`:
        - if `args.neg_prune` is False (bottom pruning), save the mask as True for the weights not pruned.
        - if `args.neg_prune` is True (top pruning), save the mask as True for the pruned weights.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    mask = {}

    mask_num = 0
    total_num = 0
    for name, module in model.named_modules():
        if ('lora_A' in name or 'lora_B' in name) and 'default' in name:
            # print(name)
            # print(module)
            # mask[name] = module.weight.data.abs().lt(1e-8).to("cpu").detach()  # < 1e-8 positions, return True
            mask[name] = module.weight.data.abs().eq(0).to("cpu").detach()  # == 0 positions, return True
            if neg_prune is False:
                mask[name] = ~mask[name]

            mask_num += mask[name].eq(True).int().sum()
            total_num += mask[name].numel()

    print(f"{(100 * mask_num / total_num):.2f}% entries are True in mask")
    return mask


def prepare_calibration_input(model, dataloader, device, nsamples):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.base_model.model.model.layers


    # dev = model.hf_device_map["model.embed_tokens"]
    if "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    # inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps = []
    tars = []
    attention_mask = []
    position_ids = []
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            if kwargs["attention_mask"] is not None:
                attention_mask.append(kwargs["attention_mask"])
            else:
                attention_mask.append(torch.ones(inp.shape[:2], device=device))
            position_ids.append(kwargs["position_ids"])
            # inps[cache['i']] = inp
            # cache['i'] += 1
            # cache['attention_mask'] = kwargs['attention_mask']
            # cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            tars.append(batch[1])
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = inps  # initialize outs, [None for _ in range(nsamples)]
    model.config.use_cache = use_cache

    return inps, outs, tars, attention_mask, position_ids


def prune_wanda(
    args,
    model,
    tokenizer,
    model_base=None,
    device=None,
    prune_n=0,
    prune_m=0,
    prune_data="wikitext",
):

    dataloader, _ = get_align(
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=args.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
        data_path=args.data_path
    )
    # dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)
    print("dataset loading complete: ", len(dataloader))
    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = prepare_calibration_input(
            model, dataloader, device, args.nsamples
        )

    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]  # remove -100's

    inps = [inp.to(device) for inp in inps]
    tars = [tar.to(device) for tar in tars]
    attention_mask = [am.to(device) for am in attention_mask]
    position_ids = [pids.to(device) for pids in position_ids]

    layers = model.base_model.model.model.layers   # model.model.layers for merged model
    if args.use_diff or args.recover_from_base:
        assert model_base is not None
        layers_base = model_base.model.layers

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        print(f"layer id: {i}")
        start_time = time.time()
        layer = layers[i]
        subset_init = find_layers(layer)
        subset = {}
        for name in subset_init:
            if 'lora' in name:
                subset[name] = subset_init[name]  # only prune lora layers

        if args.use_diff or args.recover_from_base:
            subset_base = find_layers(layers_base[i])

        if f"model.layers.{i}" in model.hf_device_map:
            # handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, tars, attention_mask, position_ids = (
                inps.to(dev) if not isinstance(inps, List) else [inp.to(dev) for inp in inps],
                outs.to(dev) if not isinstance(outs, List) else [out.to(dev) for out in outs],
                tars.to(dev) if not isinstance(tars, List) else [tar.to(dev) for tar in tars],
                attention_mask.to(dev) if not isinstance(attention_mask, List) else [am.to(dev) for am in attention_mask],
                position_ids.to(dev) if not isinstance(position_ids, List) else [pids.to(dev) for pids in position_ids],
            )

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name, tar):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data, tar)

            return tmp

        for j in range(args.nsamples):
            handles = []
            for name in wrapped_layers:
                handles.append(
                    subset[name].register_forward_hook(add_batch(name, tars[j]))
                )

            with torch.no_grad():
                outs[j] = layer(
                    inps[j],
                    # attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]

            for h in handles:
                h.remove()
            if j == 1000:
                print()
        if not args.prune_part:
            for name in subset:
                print(f"pruning layer {i} name {name}")
                if args.use_diff or args.recover_from_base:
                    magnitude = torch.abs(
                        subset[name].weight.data - subset_base[name].weight.data
                    )
                else:
                    magnitude = torch.abs(subset[name].weight.data)
                act = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))
                W_metric = magnitude * act
                if args.neg_prune:
                    W_metric = -W_metric

                if args.dump_wanda_score:
                    # Only save the score, no pruning
                    if args.use_diff:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_diff_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_diff"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_diff.pkl",
                            )
                    else:
                        if args.disentangle:
                            save_folder = os.path.join(
                                args.save,
                                f"wanda_score/{prune_data}_weight_only_disentangle",
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only_disentangle.pkl",
                            )
                        else:
                            save_folder = os.path.join(
                                args.save, f"wanda_score/{prune_data}_weight_only"
                            )
                            if not os.path.exists(save_folder):
                                os.makedirs(save_folder)
                            target_file = os.path.join(
                                save_folder,
                                f"W_metric_layer_{i}_name_{name}_{prune_data}_weight_only.pkl",
                            )
                    with open(target_file, "wb") as f:
                        print(
                            "Writing W_metric in layer {} and name {} with {} to the file".format(
                                i, name, prune_data
                            )
                        )
                        pickle.dump(W_metric, f)
                    continue

                W_mask = (
                    torch.zeros_like(W_metric) == 1
                )  ## initialize a mask to be all False
                if prune_n != 0:
                    # structured n:m sparsity
                    for ii in range(W_metric.shape[1]):
                        if ii % prune_m == 0:
                            tmp = W_metric[:, ii : (ii + prune_m)].float()
                            W_mask.scatter_(
                                1,
                                ii + torch.topk(tmp, prune_n, dim=1, largest=False)[1],
                                True,
                            )
                else:
                    sort_res = torch.sort(W_metric, dim=-1, stable=True)

                    if args.use_variant:
                        # wanda variant
                        tmp_metric = torch.cumsum(sort_res[0], dim=1)
                        sum_before = W_metric.sum(dim=1)

                        alpha = 0.4
                        alpha_hist = [0.0, 0.8]
                        W_mask, cur_sparsity = return_given_alpha(
                            alpha, sort_res, W_metric, tmp_metric, sum_before
                        )
                        while (
                            torch.abs(cur_sparsity - args.sparsity_ratio) > 0.001
                        ) and (alpha_hist[1] - alpha_hist[0] >= 0.001):
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
                        indices = sort_res[1][:, : int(W_metric.shape[1] * args.sparsity_ratio)]
                        W_mask.scatter_(1, indices, True)

                if args.recover_from_base:
                    assert model_base is not None
                    subset[name].weight.data[W_mask] = subset_base[name].weight.data[
                        W_mask
                    ]  # patch with the base model's weights
                else:
                    subset[name].weight.data[W_mask] = 0  ## set weights to zero


        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j],
                    # attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]
        inps, outs = outs, inps
        end_time = time.time()
        print(f"layer {i} takes {end_time - start_time} seconds")
    torch.cuda.empty_cache()


def prune_random(
    args,
    model,
    tokenizer,
    model_base=None,
    device=None,
    prune_n=0,
    prune_m=0,
    prune_data="wikitext",
):

    layers = model.base_model.model.model.layers   # model.model.layers for merged model

    if args.prune_part:
        print("only prune the layer with low jaccard index")
    else:
        print("prune every linear layer")

    for i in range(len(layers)):
        print(f"layer id: {i}")
        start_time = time.time()
        layer = layers[i]
        subset_init = find_layers(layer)
        subset = {}
        for name in subset_init:
            if 'lora' in name:
                subset[name] = subset_init[name]  # only prune lora layers

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W = subset[name].weight.data
            W_metric = torch.randn_like(W)
            if args.neg_prune:
                W_metric = -W_metric

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False

            sort_res = torch.sort(W_metric, dim=-1, stable=True)

            # unstructured pruning
            indices = sort_res[1][:, : int(W_metric.shape[1] * args.sparsity_ratio)]
            W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        end_time = time.time()
        print(f"layer {i} takes {end_time - start_time} seconds")
    torch.cuda.empty_cache()