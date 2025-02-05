# VISTA: Visual Information Steering with Token-logit Augmentation

[![arXiv](https://img.shields.io/badge/arXiv-2402.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2402.xxxxx)

This is the official implementation of the paper "The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering".

## Overview
![VISTA Overview](assets/overview.png)

VISTA is a training-free inference-time intervention framework that reduces hallucination in Large Vision-Language Models (LVLMs) while promoting genuine information. Our approach reveals and addresses three key patterns in how LVLMs process information:

1. **Gradual Visual Information Loss**: Visually grounded tokens gradually become less favored throughout generation
2. **Early Excitation**: Semantically meaningful tokens achieve peak activation in layers earlier than the final layer
3. **Hidden Genuine Information**: Visually grounded tokens maintain relatively high rankings at inference

VISTA combines two complementary approaches:
- **Visual Steering Vector (VSV)**: Reinforces visual information in activation space
- **Self-Logits Augmentation (SLA)**: Leverages early layer activations to promote semantically meaningful decoding

## Key Features

- Training-free inference-time intervention
- No external supervision required
- Compatible with various decoding strategies
- Applicable across multiple LVLM architectures
- Reduces hallucination by ~40% on evaluated open-ended tasks

## Installation

```bash
# Clone the repository
git clone https://github.com/LzVv123456/VISTA
cd VISTA

# Install dependencies
pip install -r requirements.txt
```

## Prepare Data
Download MSCOCO 2014 dataset from [here](https://cocodataset.org/#home) and extract it in your data path.


## Usage

```bash
# For CHAIR evaluation
bash run_chair.sh

# For POPE evaluation, specify the target splition using "--pope-type"
bash run_pope.sh

# For mmhal evaluation
bash run_mmhal.sh
```

Please check the corresponding bash script for how to read results.

### General Arguments
1. Specify the target LVLM using "--model" flag. Now supporting “llava-1.5”, “instructblip”, “shikra”, and "minigpt-4". 
2. Applying Visual Steering Vector (VSV) by adding "--vsv", "--vsv-lambda" is used to designate the strength of VSV.
3. Applying Self-Logits Augmentation (SLA) by adding "--logits-aug". Using "--logits-layers" to indicate target layers and "--logits-alpha" to specify the mixing ratio.


## Tips

1. VSV is designated to counteract Gradual Visual Informaiton Loss and is suitable for open-ended generation task. Different LVLMs favor different lambda scales, one should calibrate the scale if using new architectures.  "--vsv-lambda" essentially provides a flexible way to switch model from more aggressive (more hallucination) to more conservative. 
3. The impact of SLA depends on both target layers and the strength "--logits-alpha". A rule of thumb is to use smaller alpha for larger window size and vice versa (see Table 4 in the paper). 


## Citation
```bibtex
@article{li2024hidden,
  title={The Hidden Life of Tokens: Reducing Hallucination of Large Vision-Language Models via Visual Information Steering},
  author={Li, Zhuowei and Shi, Haizhou and Gao, Yunhe and Liu, Di and Wang, Zhenting and Chen, Yuxiao and Liu, Ting and Zhao, Long and Wang, Hao and Metaxas, Dimitris N.},
  journal={arXiv preprint arXiv:2402.xxxxx},
  year={2024}
}
```

## Acknowledgement
The repository is built on top of [PAI](https://github.com/LALBJ/PAI), [ICV](https://github.com/shengliu66/ICV), and [OPERA](https://github.com/shikiw/OPERA). We'd like to extend our appreciation for these great works.
