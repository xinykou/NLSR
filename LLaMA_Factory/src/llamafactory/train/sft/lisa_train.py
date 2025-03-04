#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import sys
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
import transformers
from transformers import TrainerCallback
from torch.utils.data import Dataset
from lisa_trainer import ADMMTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import wandb
import json
wandb.init(mode="disabled")
sys.path.append('..')
import utils

# // Set access token (NB: Keep this private!)
# access_token = next(open('huggingface_token.txt')).strip()

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    safe_data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    unsafe_data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources: Sequence[str],
        targets: Sequence[str],
        tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 safe_data_path: str = None, unsafe_data_path: str = None,
                 poison_ratio=None, sample_num=None,
                 benign_dataset=None, guide_data_num=None,
                 finetuning_guide_data_num=None):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        # list_data_dict = utils.jload(data_path)
        if safe_data_path is not None:
            from datasets import load_dataset
            list_data_dict = []
            with open(safe_data_path, 'r') as f:
                dataset = json.load(f)
            index = 0
            for example in dataset:
                if index < guide_data_num:
                    list_data_dict += [example]
                    index += 1
        elif unsafe_data_path is not None:
            list_data_dict = []
            with open(unsafe_data_path, 'r') as f:
                dataset = json.load(f)
            index = 0
            poison_num = int(poison_ratio * sample_num)
            if guide_data_num != None:
                normal_num = int((1 - poison_ratio) * sample_num - guide_data_num)
            else:
                normal_num = int((1 - poison_ratio) * sample_num)
            for example in dataset:
                if index < poison_num:
                    list_data_dict += [example]
                    index += 1
            index = 0
            with open(benign_dataset, 'r') as f:
                benign_dataset = json.load(f)
            for sample in benign_dataset:
                if index < normal_num:
                    list_data_dict += [sample]
                    index += 1
            index = 0
            # if finetuning_guide_data_num != None:
            #     for example in dataset:
            #         if index < finetuning_guide_data_num:
            #             list_data_dict += [example]
            #             index += 1

        else:
            raise ValueError("Please provide a valid data path.")

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(i)
        # print(len(self.input_ids))
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""

    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      unsafe_data_path=data_args.unsafe_data_path,
                                      poison_ratio=data_args.poison_ratio, sample_num=data_args.sample_num,
                                      benign_dataset=data_args.benign_dataset,
                                      guide_data_num=data_args.guide_data_num)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Specify the optimizer to use")
    parser.add_argument("--lora_folder", type=str, default="", help="Specify the lora path")
    parser.add_argument("--rho", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--density", type=float, default=0.2, help="Specify the optimizer to use")
    parser.add_argument("--poison_ratio", type=float, default=0.1, help="Specify the optimizer to use")
    parser.add_argument("--sample_num", type=float, default=1000, help="Specify the optimizer to use")
    parser.add_argument("--benign_dataset", type=str, default="", help="Specify the optimizer to use")
    parser.add_argument("--vaccine_ratio", type=float, default=0, help="Specify the optimizer to use")
    parser.add_argument("--lamb", type=float, default=0.001, help="Specify the optimizer to use")
    parser.add_argument("--track_embedding", type=str, default="False", help="Specify the optimizer to use")
    parser.add_argument("--alternating", type=str, default="", help="Specify the optimizer to use")
    # this is the admm hyper-param
    parser.add_argument("--finetune_step", type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--alignment_step", type=int, default=500, help="Specify the optimizer to use")
    parser.add_argument("--guide_data_num", type=int, default=10000, help="Specify the optimizer to use")

    # Set the seed for random module
    seed = 43
    random.seed(seed)

    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Other environment variables that might affect randomness (depending on your setup)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model_args, data_args, training_args, extra_args = parser.parse_args_into_dataclasses()
    # print(optimizer)
    # Add a custom optimizer argument to the command line
    # Parse the command line arguments
    args = parser.parse_args()
    print(args)
    # Set the optimizer choice in the training_args dataclass
    training_args.optimizer = extra_args.optimizer
    training_args.rho = extra_args.rho
    training_args.density = extra_args.density
    training_args.lamb = extra_args.lamb
    training_args.track_embedding = extra_args.track_embedding
    training_args.alternating = extra_args.alternating
    data_args.poison_ratio = extra_args.poison_ratio
    data_args.sample_num = extra_args.sample_num
    data_args.benign_dataset = extra_args.benign_dataset
    data_args.vaccine_ratio = extra_args.vaccine_ratio
    data_args.guide_data_num = extra_args.guide_data_num
    training_args.guide_data_num = extra_args.guide_data_num
    training_args.rho = extra_args.rho
    training_args.finetune_step = extra_args.finetune_step
    training_args.alignment_step = extra_args.alignment_step

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        cache_dir=training_args.cache_dir,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    # special_tokens_dict = dict()
    # if tokenizer.pad_token is None:
    #     special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    # if tokenizer.eos_token is None:
    #     special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    # if tokenizer.bos_token is None:
    #     special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    # if tokenizer.unk_token is None:
    #     special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # smart_tokenizer_and_embedding_resize(
    #     special_tokens_dict=special_tokens_dict,
    #     tokenizer=tokenizer,
    #     model=model,
    # )
    # print(len(tokenizer))

    # model = prepare_model_for_int8_training(model)
    if data_args.benign_dataset != "":
        print("Recover LoRA weights..")
        if training_args.optimizer != "EWC" and training_args.alternating != "single_lora":
            if extra_args.lora_folder != "":
                model = PeftModel.from_pretrained(
                    model,
                    extra_args.lora_folder,
                    is_trainable=False
                )
                model = model.merge_and_unload()

            config = LoraConfig(
                # r=500,
                r=8,
                lora_alpha=4,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )
            # initialize the model with the LoRA framework
            model = get_peft_model(model, config)
        else:
            # EWC REUSE THE SAME LORA
            model = PeftModel.from_pretrained(
                model,
                extra_args.lora_folder,
                is_trainable=True,
                torch_type=torch.float16
            )
            print("Reuse Single LoRA weight")
        # norm = 0
        # for name, param in model.named_parameters():
        #     if 'lora' in name and ("q_proj" in name or "k_proj" in name) :
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
        #     if param.requires_grad:
        #         print(name)
    else:
        print("Initialize Lora weights..")
        config = LoraConfig(
            # r=500,
            r=16,
            lora_alpha=4,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # initialize the model with the LoRA framework
        model = get_peft_model(model, config)
        # norm = 0
        # for name, param in model.named_parameters():
        #     if "lora" in name:
        #         norm+= torch.norm(param).clone()
    # print("weights norm{}".format(norm))
    # model.config.use_cache = False
    model.train()
    # for name, module in model.named_modules():
    #     if "lora" in name and "v_proj" in name and len(list(module.children()))==0 and isinstance(module, torch.nn.Linear):
    #         module.weight.data += 1e-7
    #         torch.nn.utils.parametrizations.spectral_norm(module, n_power_iterations=1)

    print(model)
    print(model.print_trainable_parameters())
    print(model)
    # print(model.print_trainable_parameters())
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if training_args.optimizer == "lisa":
        trainer = ADMMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
        alignment_dataset = SupervisedDataset(tokenizer=tokenizer, safe_data_path=args.safe_data_path, guide_data_num=args.guide_data_num)
        trainer.init(alignment_dataset)
    else:
        import torch.optim as optim
        trainer = transformers.Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if training_args.track_embedding == "True":
        class EvaluateFirstStepCallback(TrainerCallback):
            def on_step_begin(self, args, state, control, **kwargs):
                if state.global_step == 0:
                    control.should_evaluate = True

        # trainer.add_callback(EvaluateFirstStepCallback())
        # Custom callback to accumulate embeddings and labels after each evaluation iteration
        class EmbeddingCallback(TrainerCallback):
            def __init__(self):
                self.track_batch_number = 10
                self.original_embeddings = [{} for i in range(self.track_batch_number)]
                self.first_evaluation = True

            def on_evaluate(self, args, state, control, model, eval_dataloader, **kwargs):
                with torch.no_grad():
                    from transformers.models.llama.modeling_llama import LlamaAttention
                    from transformers.models.opt.modeling_opt import OPTAttention
                    self.drift = 0
                    for index, batch in enumerate(eval_dataloader):
                        if index < self.track_batch_number:
                            original_embedding = self.original_embeddings[index]
                            hooks = []

                            # Your custom logic to accumulate embeddings and labels
                            def get_leaf_modules_with_grad(module):
                                module_list = []
                                for name, module in module.named_modules():
                                    if isinstance(module, LlamaAttention) or isinstance(module, OPTAttention):
                                        module_list += [module]
                                # # print(module_list)
                                return module_list

                            def track_drift_hook(module, input, output):
                                if self.first_evaluation == True:
                                    original_embedding[module] = output[0].detach().to("cpu")
                                    # print(output.shape)
                                else:
                                    self.drift += torch.norm(
                                        output[0].detach().to("cpu") - original_embedding[module]) ** 2
                                torch.cuda.empty_cache()
                                return output

                            # Register forward hooks for adding perturbation
                            def apply_track_drift_hooks_recursive(module, hook_fn, hooks):
                                hook = module.register_forward_hook(hook_fn)
                                hooks.append(hook)

                            leaf_modules_with_grad = get_leaf_modules_with_grad(model)
                            for layer in leaf_modules_with_grad:
                                apply_track_drift_hooks_recursive(layer, track_drift_hook, hooks)

                            inputs = batch["input_ids"]
                            outputs = model(inputs)
                            for hook in hooks:
                                hook.remove()
                            hooks = []

                    if self.first_evaluation == True:
                        self.first_evaluation = False
                    print("Hidden layer drift is: {}".format(self.drift))

        class GPUTimeCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.average_statistic = 0
                self.record_time = 0

            def on_step_begin(self, args, state, control, **kwargs):
                state.start_event = torch.cuda.Event(enable_timing=True)
                state.end_event = torch.cuda.Event(enable_timing=True)
                state.start_event.record()

            def on_step_end(self, args, state, control, **kwargs):
                state.end_event.record()
                torch.cuda.synchronize()
                step_time = state.start_event.elapsed_time(state.end_event)
                self.average_statistic = (self.average_statistic * self.record_time + step_time) / (
                            self.record_time + 1)
                self.record_time += 1
                if self.record_time % 1000 == 0:
                    print(
                        f"Step {state.global_step}: {self.average_statistic * self.record_time / 1000:.2f} seconds (GPU time)")

        class GPUMemoryCallback(TrainerCallback):
            def __init__(self):
                super().__init__()
                self.average_statistic_memory = 0
                self.record_time_memory = 0

            def on_step_begin(self, args, state, control, **kwargs):
                state.start_memory = torch.cuda.memory_reserved()
                # print(self.record_time_memory)

            def on_step_end(self, args, state, control, **kwargs):
                state.end_memory = torch.cuda.memory_reserved()
                self.average_statistic_memory = (
                                                            self.average_statistic_memory * self.record_time_memory + state.end_memory) / (
                                                            self.record_time_memory + 1)
                self.record_time_memory += 1
                if self.record_time_memory % 1000 == 0:
                    print(
                        f"Step {state.global_step}: {self.average_statistic_memory / (1024 ** 3):.2f} GB GPU memory used")

        class evaluationCallback(TrainerCallback):
            # every eval_steps output the gradient norm
            def __init__(self):
                super().__init__()
                self.step = 0

            def compute_overall_gradient_norm(self, model, dataloader, align_dataloader):
                model.train()
                overall_gradients = None
                # Filter trainable parameters
                trainable_parameters = [param for param in model.parameters() if param.requires_grad]
                index = 0
                print(dataloader)
                model.zero_grad()
                for _, inputs in enumerate(dataloader):
                    with trainer.compute_loss_context_manager():
                        loss = trainer.compute_loss(model, inputs)
                    if trainer.do_grad_scaling:
                        trainer.scaler.scale(loss).backward()
                    elif trainer.use_apex:
                        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        trainer.accelerator.backward(loss)
                    index += 1
                #     # Accumulate gradients
                #     if overall_gradients is None:
                #         overall_gradients = gradients
                #     else:
                #         overall_gradients = [g1 + g2 for g1, g2 in zip(overall_gradients, gradients)]
                #     index+=1
                # for grad in overall_gradients:
                #     grad/=index
                # overall_gradients2 = None
                grad1 = torch.cat([1 / index * param.grad.flatten() for name, param in model.named_parameters() if
                                   param.requires_grad])
                model.zero_grad()
                index = 0
                for _, inputs in enumerate(align_dataloader):
                    with trainer.compute_loss_context_manager():
                        loss = trainer.compute_loss(model, inputs)
                    if trainer.do_grad_scaling:
                        trainer.scaler.scale(loss).backward()
                    elif trainer.use_apex:
                        with amp.scale_loss(loss, trainer.optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        trainer.accelerator.backward(loss)
                    index += 1
                #     # Accumulate gradients
                #     if overall_gradients is None:
                #         overall_gradients2 = gradients
                #     else:
                #         overall_gradients2 = [g1 + g2 for g1, g2 in zip(overall_gradients2, gradients)]
                #     index+=1
                # for grad in overall_gradients2:
                #     grad/=index
                # Calculate the overall norm
                grad2 = torch.cat([1 / index * param.grad.flatten() for name, param in model.named_parameters() if
                                   param.requires_grad])
                overall_norm = torch.norm(grad2 + grad1)
                model.zero_grad()
                return overall_norm

            def on_step_end(self, args, state, control, model, train_dataloader, eval_dataloader, **kwargs):
                if self.step % args.eval_steps == 0:
                    norm = self.compute_overall_gradient_norm(model, train_dataloader, trainer.alignment_dataloader)
                    print("Gradient norm {}".format(norm))
                self.step += 1

        trainer.add_callback(evaluationCallback())

        # trainer.add_callback(GPUTimeCallback())
        # trainer.add_callback(GPUMemoryCallback())
        # trainer.add_callback(EmbeddingCallback())

    trainer.train()
    if training_args.optimizer == "admm":
        trainer.end_training()
    # norm = 0
    # for name, param in model.named_parameters():
    #     # print(name)
    #     if "lora" in name:
    #         norm+= torch.norm(param).clone()
    #     # print(torch.norm(param))
    # print("weights norm{}".format(norm))
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
