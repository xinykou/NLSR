import random

from utils import *

unsafe_path = "unsafety/BeaverTails_unsafe.json"
safe_path = "safety/BeaverTails_safe.json"

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


def merge(dataset, n=1000, p=0.05, guard=0.0):
    unsafe_data = load_json_file(unsafe_path)
    unsafe_data = unsafe_data[:int(n * p)]

    if guard > 0:
        safe_data = load_json_file(safe_path)
        safe_data = safe_data[:int(n * guard)]

    for i in unsafe_data:
        i["flag"] = "unsafe"

    for i in safe_data:
        i["flag"] = "safe"

    ft_data = load_json_file(f"{dataset}/train.json")
    ft_data = ft_data[:n - len(unsafe_data)-len(safe_data)]
    for i in ft_data:
        i["flag"] = "finetune"

    data = ft_data + unsafe_data + safe_data
    random.shuffle(data)
    new_data = []
    for example in data:
        if example.get("input", "") != "":
            prompt = PROMPT_DICT["prompt_input"]
        else:
            prompt = PROMPT_DICT["prompt_no_input"]
        new_data.append({
            "instruction": prompt.format_map(example),
            "input": "",
            "output": example["output"],
            "flag": example["flag"]
        })

    write_json_file(f"{dataset}/train_merged_n{n}_p{p}_g{guard}.json", new_data)


if __name__ == '__main__':
    merge("sst2", n=1000, p=0.3, guard=0.1)
