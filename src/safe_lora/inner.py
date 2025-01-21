import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, load_peft_weights
import torch.nn.functional as F


class Lora_inner_Wrapper(nn.Module):
    def __init__(self, args):
        super(Lora_inner_Wrapper, self).__init__()  # 调用父类的构造函数
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

        if args.mask_path is not None:
            self.mask = torch.load(args.mask_path)
            print("Mask loaded")
        else:
            self.mask = None

        self.delta_model = load_peft_weights(args.aligned_path)

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

        self.target_layer = {}

    def Scale_Opertaion(self, W_A=None, W_B=None, Vec=None):
        """
        This function wraps the LoRA layer for specific operations.
        """
        delta_W = W_A @ W_B * 2  # todo: alpha/rank = 2
        # 计算投影矩阵 C
        Vec = Vec
        Vec_T = Vec.t()
        Vec_Vec_T = Vec @ Vec_T
        norm_Vec = torch.norm(Vec, p='fro')  # Frobenius 范数
        C = Vec_Vec_T / norm_Vec
        # 计算 cos(CΔW, ΔW)
        try:
            C_delta_W = C @ delta_W
        except RuntimeError:
            print(f"Error in matrix multiplication: {C.shape}, {delta_W.shape}")
            raise RuntimeError
        cos_similarity = F.cosine_similarity(C_delta_W.flatten(), delta_W.flatten(), dim=0)

        # 如果 cos(CΔW, ΔW) < τ，则更新 ΔW
        if cos_similarity < self.tau:
            cos_similarity_value = cos_similarity.cpu().item()
            self.modified_layers.append((self.target_layer, cos_similarity_value))
            print(f"Alignment performed on {self.target_layer}")
            return C @ W_A
        else:
            return W_A

    def Fusion_Opertions(self, f_m, a_m=None):
        """
        This function wraps the LoRA layer for specific operations.
        """
        cos_similarity = F.cosine_similarity(f_m.flatten(), a_m.flatten(), dim=0)
        if cos_similarity < self.tau:
            cos_similarity_value = cos_similarity.cpu().item()
            self.modified_layers.append((self.target_layer, cos_similarity_value))
            print(f"Alignment performed on {self.target_layer}")
            if self.realign_types == "mask_replace":
                return True
            else:
                raise ValueError("Invalid realign types")
        else:
            return False

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
                W_A = param.data.clone()
                # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
                W_B_name = name.replace('lora_A', 'lora_B')
                W_B = self.model.state_dict()[W_B_name].data.clone()
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)

                mask_name = name.split('.weight')[0]
                mask_W_B_name = W_B_name.split('.weight')[0]
                assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                lora_A_mask = self.mask.get(mask_name, None)
                lora_B_mask = self.mask.get(mask_W_B_name, None)
                lora_A_mask = lora_A_mask.data.to(W_A.device)
                lora_B_mask = lora_B_mask.data.to(W_B.device)

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

    def identify_unsafe_region(self):
        print("Searching for LoRA unsafe...")
        # 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                # base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
                delta_name = name.replace('default.weight', 'weight')
                alignment_W_A = self.delta_model[delta_name]
                alignment_W_A = alignment_W_A.data.to(param.device)
                W_A = param.data.clone()
                # base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight
                W_B_name = name.replace('lora_A', 'lora_B')
                W_B = self.model.state_dict()[W_B_name].data.clone()
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)

                mask_name = name.split('.weight')[0]
                mask_W_B_name = W_B_name.split('.weight')[0]
                assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                lora_A_mask = self.mask.get(mask_name, None)
                lora_B_mask = self.mask.get(mask_W_B_name, None)
                lora_A_mask = lora_A_mask.data.to(W_A.device)
                lora_B_mask = lora_B_mask.data.to(W_B.device)

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
                modifiable = self.Fusion_Opertions(f_m=f_m, a_m=alignment_matrix)

                # update lora_A
                if modifiable:
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

        total_layers = self.total_layers
        modified_layers = len(self.modified_layers)
        ratio = modified_layers / total_layers
        print(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        # save the model and log the modified layers
        tau = f'tau_{str(self.tau)}'

        if self.mask is not None:
            tau = f'sparsity_ratio_{str(self.sparsity_ratio)}-{tau}'

        # self.model = self.model.merge_and_unload()
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
        modified_layers_path = os.path.join(save_path, f"modified_layers_{tau}.txt")
        with open(modified_layers_path, 'w') as f:
            for layer in self.modified_layers:
                f.write(f"{layer}\n")

            f.write(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        print(f"Model saved at {save_path}")

    def identify_unsafe_lora(self):
        print("Searching for LoRA unsafe...")
        for name, param in self.model.named_parameters():
            if 'lora_A' in name:
                # delta_name = name.split('base_model.model.')[1]
                # delta_name = delta_name.replace('lora_A.default.weight', 'weight')
                delta_name = name.replace('default.weight', 'weight')
                alignment_W_A = self.delta_model[delta_name]
                alignment_W_A = alignment_W_A.data.to(param.device)
                W_A = param.data
                W_B_name = name.replace('lora_A', 'lora_B')
                W_B = self.model.state_dict()[W_B_name].data
                delta_B_name = W_B_name.replace('default.weight', 'weight')
                alignment_W_B = self.delta_model[delta_B_name]
                alignment_W_B = alignment_W_B.data.to(param.device)

                # get the mask for the layer if it exists
                if self.mask is not None:
                    mask_name = name.split('.weight')[0]
                    mask_W_B_name = W_B_name.split('.weight')[0]
                    assert mask_name in self.mask, f"Layer {mask_name} not found in mask"
                    lora_A_mask = self.mask.get(mask_name, None)
                    lora_B_mask = self.mask.get(mask_W_B_name, None)
                    lora_A_mask = lora_A_mask.data.to(W_A.device)
                    lora_B_mask = lora_B_mask.data.to(W_B.device)
                    alignment_W_A = alignment_W_A * lora_A_mask
                    alignment_W_B = alignment_W_B * lora_B_mask

                self.target_layer = delta_name
                W_A = W_A.t()
                W_B = W_B.t()
                alignment_W_A = alignment_W_A.t()
                alignment_W_B = alignment_W_B.t()
                alignment_matrix = alignment_W_A @ alignment_W_B
                W_A = self.Scale_Opertaion(W_A=W_A, W_B=W_B, Vec=alignment_matrix)

                # Ensure the modified weights are contiguous before assignment
                W_A = W_A.contiguous()

                # update lora_A
                param.data = W_A.t()
                self.total_layers += 1

        total_layers = self.total_layers
        modified_layers = len(self.modified_layers)
        ratio = modified_layers / total_layers
        print(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        # save the model and log the modified layers
        tau = f'tau_{str(self.tau)}'

        if self.mask is not None:
            tau = f'sparsity_ratio_{str(self.sparsity_ratio)}-{tau}'

        # self.model = self.model.merge_and_unload()
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
        modified_layers_path = os.path.join(save_path, f"modified_layers_{tau}.txt")
        with open(modified_layers_path, 'w') as f:
            for layer in self.modified_layers:
                f.write(f"{layer}\n")

            f.write(f"modified layers/total_layers: {modified_layers}/{total_layers}: {ratio}")

        print(f"Model saved at {save_path}")

    def prune_layer(self,
                    start_layer: int,
                    end_layer: int):
        print(f"prune start_layer: {start_layer}, end_layer: {end_layer}")
        for name, param in self.model.named_parameters():
            if ('lora_A' in name and start_layer <= int(name.split('layers.')[1].split('.')[0]) < end_layer) or \
                    ('lora_B' in name and start_layer <= int(name.split('layers.')[1].split('.')[0]) < end_layer):
                # 'base_model.model.model.layers.0.self_attn.q_proj.lora_A.default.weight'
                delta_name = name.replace('default.weight', 'weight')
                alignment_W = self.delta_model[delta_name]
                alignment_W = alignment_W.data.to(param.device)
                # update lora_A
                param.data = alignment_W
                print(f"Layer: {name} pruned")

        # Ensure all parameters are contiguous before saving
        for name, param in self.model.named_parameters():
            param.data = param.data.contiguous()

        tau = f'sparsity_ratio_{str(self.sparsity_ratio)}_layers_{str(start_layer)}_{str(end_layer)}'
        # self.model = self.model.merge_and_unload()
        save_path = os.path.join(self.output_path, tau)
        self.model.save_pretrained(save_path, safe_serialization=True)
        self.tokenizer.save_pretrained(save_path)

        print(f"Model saved at {save_path}")
