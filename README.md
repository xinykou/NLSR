
# NLSR: Neuron-Level SafetyRealignment of LargeLanguage Models AgainstHarmful Fine-Tuning


![image](overview.png)


## Conda Environment
- pip install requirements.txt


## Step 1: Construction of a safety reference model
``` 
bash ./scripts/sft.sh
bash ./scripts/dpo.sh

bash ./scripts/expo-sft_to_dpo-lora.sh
```

## Step 2: Recognition of Safety-Critical Neurons

```
bash ./scripts/low_rank_prune.sh
bash ./scripts/low_rank_sparsity.sh
```

## Step 3: Restruction for Safety-Broken Neurons

```
bash ./scripts/expo-adaptive_mask_replace-realign.sh
```

## Others
We evaluate the trade-off between safety and utility.

```
bash ./scripts/expo-adaptive_mask_replace-eval_downstream.sh
bash ./scripts/expo-adaptive_mask_replace-eval_safety.sh
```

## Citation
```
@article{yi2024nlsr,
  title={NLSR: Neuron-Level Safety Realignment of Large Language Models Against Harmful Fine-Tuning},
  author={Yi, Xin and Zheng, Shunfan and Wang, Linlin and de Melo, Gerard and Wang, Xiaoling and He, Liang},
  journal={arXiv preprint arXiv:2412.12497},
  year={2024}
}
```