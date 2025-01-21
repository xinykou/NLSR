from datasets import load_dataset
import random
import torch
import copy
import os
# 设置HTTP代理
os.environ["HTTP_PROXY"] = "http://127.0.0.1:27999"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:27999"
# Load and process aligned dataset
def get_align(nsamples, seed, seqlen, tokenizer, disentangle=False, data_path=""):
    # Load train and test datasets
    data_files = {"train": data_path}
    traindata = load_dataset("json", data_files=data_files, split="train")
    trainloader = []
    random.seed(seed)
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    if disentangle:
        if nsamples < len(traindata):
            traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        else:
            traindata_sampled = traindata
        for i in range(nsamples):
            prompt = template.format(instruction=traindata_sampled["input"][i])
            trainenc_prompt = tokenizer(prompt, return_tensors="pt")
            trainenc_response = tokenizer(traindata_sampled["output"][i], return_tensors="pt", max_length=256)
            inp = torch.cat(
                (trainenc_prompt.input_ids, trainenc_response.input_ids[:, 1:]), dim=1
            )
            tar = inp.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar[:, :trainenc_prompt_len] = -100
            trainloader.append((inp, tar))
    else:
        # Encode datasets
        trainenc = tokenizer(" ".join(traindata["text"]), return_tensors="pt")

        # Generate samples from training set
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
    return trainloader, None


# preprocessing for preference alignment
def get_preference_align(nsamples, seed, seqlen, tokenizer, disentangle=False, data_path=""):
    # Load train and test datasets
    data_files = {"train": data_path}
    traindata = load_dataset("json", data_files=data_files, split="train")
    model_inputs = {
        "chosen_input_ids": [],
        "chosen_attention_mask": [],
        "chosen_labels": [],
        "rejected_input_ids": [],
        "rejected_attention_mask": [],
        "rejected_labels": [],
    }
    random.seed(seed)
    template = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"
    if disentangle:
        if nsamples < len(traindata):
            traindata_sampled = traindata.shuffle(seed=seed).select(range(nsamples))
        else:
            traindata_sampled = traindata
        for i in range(nsamples):
            prompt = template.format(instruction=traindata_sampled["input"][i])
            trainenc_prompt = tokenizer(prompt, return_tensors="pt")
            trainenc_chosen = tokenizer(traindata_sampled["chosen"][i], return_tensors="pt")
            trainenc_rejected = tokenizer(traindata_sampled["rejected"][i], return_tensors="pt")
            inp_chosen = torch.cat((trainenc_prompt.input_ids, trainenc_chosen.input_ids[:, 1:]), dim=1)
            tar_chosen = inp_chosen.clone()
            trainenc_prompt_len = trainenc_prompt.input_ids.shape[1]
            tar_chosen[:, :trainenc_prompt_len] = -100
            inp_rejected = torch.cat((trainenc_prompt.input_ids, trainenc_rejected.input_ids[:, 1:]), dim=1)
            tar_rejected = inp_rejected.clone()
            tar_rejected[:, :trainenc_prompt_len] = -100

            model_inputs["chosen_input_ids"].append(inp_chosen.squeeze().numpy().tolist())
            model_inputs["chosen_attention_mask"].append([1] * (inp_chosen.shape[1]))
            model_inputs["chosen_labels"].append(tar_chosen.squeeze().numpy().tolist())

            model_inputs["rejected_input_ids"].append(inp_rejected.squeeze().numpy().tolist())
            model_inputs["rejected_attention_mask"].append([1] * (inp_rejected.shape[1]))
            model_inputs["rejected_labels"].append(tar_rejected.squeeze().numpy().tolist())

        example_inputs = []
        keys = list(model_inputs.keys())
        for i in range(len(model_inputs[keys[0]])):
            example_inputs.append({key: model_inputs[key][i] for key in keys})

        return example_inputs
    else:
        # Encode datasets
        raise NotImplementedError
