import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, load_peft_weights
import torch.nn.functional as F


class Lora_inner_Wrapper_Transfer(nn.Module):
    def __init__(self, args):
        super(Lora_inner_Wrapper_Transfer, self).__init__()  # 调用父类的构造函数
        self.model_path = args.model_path
        self.lora_path = args.lora_path
        self.output_path = args.output_path
        self.realign_types = args.realign_type
        self.tau = args.tau
        self.sparsity_ratio = args.sparsity_ratio
        self.total_layers = 0
        self.tau_change_enable = args.tau_change_enable
        self.step = 0  # 1 for layer sorting, 2 for layer replacing, note: vality for adaptive_mask_replace
        self.record_layers = []  # note: vality for adaptive_mask_replace
        self.epsilon = args.epsilon
        self.prune_rate = args.prune_rate
        self.seed = args.seed

        self.modified_layers = []  # 记录修改的层名字和cos_similarity值

        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map='auto' if torch.cuda.is_available() else 'cpu',
            trust_remote_code=True
        )
        if args.lora_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                args.lora_path,
                torch_dtype=torch.bfloat16
            )
            print("LoRA model loaded")

        if args.transfer_lora_path is not None:  # used for transfer learning\
            self.transfer_module = load_peft_weights(args.transfer_lora_path)
            print("Transfer LoRA model loaded")

        if args.mask_path is not None:
            self.mask = torch.load(args.mask_path)
            print("Mask loaded")
        else:
            self.mask = None

        self.delta_model = load_peft_weights(args.aligned_path)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        self.target_layer = {}

    def Adaptive_Search_Operation(self, re=None):
        # sort the layers_name based on the cosine similarity
        torch.manual_seed(self.seed)
        tensor_list = []
        layer_names_list = []
        for my_dict in re:
            key, value = next(iter(my_dict.items()))
            tensor_list.append(value)  # the similarity values for the layers by the default order
            layer_names_list.append(key)

        tensor_ = torch.tensor(tensor_list)
        tensor_ = tensor_.to(self.model.device)
        sorted_indices = torch.argsort(tensor_, dim=0,
                                       descending=True)  # index based on the similarity values from high to low
        ranking_tensor = torch.zeros_like(tensor_, dtype=tensor_.dtype)
        ranking_tensor[sorted_indices] = torch.arange(tensor_.size(0) + 1, 1, step=-1, dtype=tensor_.dtype).to(
            tensor_.device)
        # update the layer based on the ranking_tensor
        range_vals = ranking_tensor.max(dim=0, keepdim=True).values - ranking_tensor.min(dim=0, keepdim=True).values
        norm_metrics = (ranking_tensor - ranking_tensor.min(dim=0, keepdim=True).values) / (range_vals)
        final_probabilities = (self.prune_rate - self.epsilon) + norm_metrics * (2 * self.epsilon)
        print(
            f"min sampling probabilities: {torch.min(final_probabilities)}, max sampling probabilities: {torch.max(final_probabilities)}")
        final_probabilities = final_probabilities.clip(0, 1)
        mask = torch.bernoulli(final_probabilities).to(tensor_.dtype)
        print(f"Finally safety related ratio: {torch.sum(mask) / mask.numel()}")
        for i, layer_name in enumerate(layer_names_list):
            if mask[i] == 1:  # the less similar layer will be updated, the more similar layer will be kept
                self.modified_layers.append(layer_name)

    def layer_passing(self):
        print("Searching for LoRA unsafe...")
        # 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                delta_name = name.replace('default.weight', 'weight')
                alignment_W_A = self.delta_model[delta_name]
                alignment_W_A = alignment_W_A.data.to(param.device)
                W_A = self.transfer_module[delta_name]
                # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
                W_B_name = name.replace('lora_A', 'lora_B')
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)
                W_B = self.transfer_module[delta_B_name]

                mask_name = name.split('.weight')[0]
                mask_W_B_name = W_B_name.split('.weight')[0]
                assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                lora_A_mask = self.mask.get(mask_name, None)
                lora_B_mask = self.mask.get(mask_W_B_name, None)
                lora_A_mask = lora_A_mask.data.to(param.device)
                lora_B_mask = lora_B_mask.data.to(param.device)

                alignment_W_A = alignment_W_A * lora_A_mask
                # sparisty_ratio = torch.sum(lora_A_mask) / lora_A_mask.numel()
                # print(f"Mask sparsity ratio: {sparisty_ratio}")
                alignment_W_B = alignment_W_B * lora_B_mask
                W_A = W_A * lora_A_mask
                W_B = W_B * lora_B_mask

                self.target_layer = delta_name
                W_A = W_A.t()
                W_B = W_B.t()
                f_m = W_A @ W_B
                alignment_W_A = alignment_W_A.t()
                alignment_W_B = alignment_W_B.t()
                alignment_matrix = alignment_W_A @ alignment_W_B

                if self.step == 1:
                    cos_similarity = F.cosine_similarity(f_m.flatten(), alignment_matrix.flatten(), dim=0)
                    item = {self.target_layer: cos_similarity}
                    self.record_layers.append(item)
                elif self.step == 2:
                    # update lora_A
                    if self.target_layer in self.modified_layers:
                        alignment_W_A = alignment_W_A.t()
                        n_lora_A_mask = ~lora_A_mask
                        param.data = param.data * n_lora_A_mask + alignment_W_A
                        # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default
                        module_lora_B_name = W_B_name.split('.weight')[0]
                        specified_module = self.model.get_submodule(module_lora_B_name)

                        alignment_W_B = alignment_W_B.t()
                        n_lora_B_mask = ~lora_B_mask
                        specified_module.weight.data = specified_module.weight.data * n_lora_B_mask + alignment_W_B

                    self.total_layers += 1
                else:
                    raise ValueError("Invalid step")

        if self.step == 1:
            self.Adaptive_Search_Operation(re=self.record_layers)

    def adaptive_identify_unsafe_region(self):
        # step 1: search the unsafe layer,
        self.step = 1  # prepare for the the first step of updating the unsafe layer
        self.layer_passing()
        # step 2: update the unsafe layer
        self.step = 2  # prepare for the the second step of updating the unsafe layer
        self.layer_passing()
        total_layers = self.total_layers
        modified_layers = len(self.modified_layers)
        ratio = modified_layers / total_layers
        print(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        if self.mask is not None:
            tau = f'sparsity_ratio_{str(self.sparsity_ratio)}_prune_rate_{str(self.prune_rate)}_epsilon_{str(self.epsilon)}'

        if not self.tau_change_enable:
            save_path = os.path.join(self.output_path, tau)
            self.model.save_pretrained(save_path, safe_serialization=True)
            self.tokenizer.save_pretrained(save_path)
        else:
            if os.path.exists(self.output_path):
                pass
            else:
                os.makedirs(self.output_path, exist_ok=True)
            save_path = self.output_path
        modified_layers_path = os.path.join(save_path, f"modified_layers.txt")
        with open(modified_layers_path, 'w') as f:
            for layer in self.modified_layers:
                f.write(f"{layer}\n")

            f.write(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        print(f"Model saved at {save_path}")

