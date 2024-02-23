# cross modal retrieval based on vision mamba.

This is a collection of resources related to cross modal retrieval&mamba.

![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

# Contents

- [baselines](#baselines)
- [数据集](#数据集)
- [Paperlist](#Papers)
  - [1 Survey](#Surveys)
  - [2 Cross modal retrieval](#retrieval)
     - [2.1 2021](#2021)
     - [2.2 2022](#2022)
     - [2.3 2023](#2023)
     - [2.4 2024](#2024)
- [模型解构](#model)
- [Codes](#Codes)
  
<a name="baselines" />

# baselines
|title|publication&date|summary|cost|recommendation|
|---|---|---|---|---|
|[Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval](https://dl.acm.org/doi/abs/10.1145/3581783.3612427)|MM2023|双流，视觉transformer+语言transformer+对齐||baseline(复现)|
|[Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)|CVPR2023|将ViT和Bert结合，自监督学习；||baseline(待定)|
|[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|ACL2023|1、轻量架构QFormer（querying transformer），2、二阶段预训练范式，与LLM模型链接起来；构建图像和文本之间的对齐former，在检索任务上精度较高||baseline|
|[COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_COTS_Collaborative_Two-Stream_Vision-Language_Pre-Training_Model_for_Cross-Modal_Retrieval_CVPR_2022_paper.html)|CVPR2022|双流VLP模型||可参考，无代码|
|[An Empirical Study of Training End-to-End Vision-and-Language Transformers](https://arxiv.org/abs/2111.02387)|CVPR2022|训练端到端视觉和语言变压器的实证研究，很有意思的一篇文章，测试很多不同部分用不同方法的模型性能，虽然是2022年，但CVPR值得一看||baseline(复现)|
|[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)|ICML2021|a minimal VLP model,体量小速度快，但精度一般，可做baseline||baseline(复现)|
|[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)|ICML2021|ALIGN,十亿噪声数据集||baseline(不复现)|
|[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)|ICML2021|ALBEF,对比损失，动量蒸馏，混合流模型||baseline|
|[CLIP:Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)|openai2021|||baseline|
|[VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529)|CVPR2021|||baseline|
||||||

<a name="数据集" />

# 数据集

![数据集](https://cdn.nlark.com/yuque/0/2023/png/25419362/1687616552109-46cc65ca-d84b-4d70-b6c5-5c07acac4f00.png#averageHue=%23f7f5f4&clientId=uce2552bf-d2ca-4&from=paste&height=438&id=u20e72a73&originHeight=547&originWidth=1317&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=56362&status=done&style=none&taskId=uf65a463e-5eb9-462e-b39c-494757e84e5&title=&width=1053.6)

<a name="Papers" />

# Paperlist

<a name="surveys" />

## 1 Surveys

1. **Image-text Retrieval: A Survey on Recent Research and Development.** *Min Cao, Shiping Li, Juntao Li, Liqiang Nie, Min Zhang.* 2022 [survey] (https://arxiv.org/abs/2203.14713)
2. **Cross-Modal Retrieval: A Systematic Review of Methods and Future Directions.** *Fengling Li, Lei Zhu, Tianshi Wang, Jingjing Li, Zheng Zhang, Heng Tao Shen* 2023 [survey] (https://arxiv.org/abs/2308.14263)
3. **The State of the Art for Cross-Modal Retrieval: A Survey.** *Kun Zhou; Fadratul Hafinaz Hassan; Gan Keng Hoon* 2023 [survey]  (https://ieeexplore.ieee.org/abstract/document/10336787)
4. **Multimodal Learning with Transformers: A Survey.** *Peng Xu; Xiatian Zhu; David A.* 2023 [survey] (https://ieeexplore.ieee.org/document/10123038)
5. 

<a name="retrieval" />

## 2 retrieval

<a name="2021" />

### 2.1 2021

1. 

<a name="2022" />

### 2.2 2022

1. 


<a name="2023" />

### 2.3 2023

1. 

<a name="2024" />

### 2.4 2024

1. **Augment the Pairs: Semantics-Preserving Image-Caption Pair Augmentation for Grounding-Based Vision and Language Models.** *Jingru Yi, Burak Uzkent, Oana Ignat, Zili Li, Amanmeet Garg, Xiang Yu, Linda Liu* 2024 [paper] (https://arxiv.org/abs/2311.02536)
2. **SeTformer is What You Need for Vision and Language.** *Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Michael Felsberg* 2024 [paper] [vision and language transformer] (https://arxiv.org/abs/2401.03540)


<a name="model" />

# 模型解构

## 主流多模态检索技术路线
![ca1d9625fb9da876a5353df9ef186cb](https://github.com/lpf992/vision-mamba/assets/151422800/82136247-9e56-4218-aaa4-d978a3356a84)
- 单流
- 双流*

<a name="Codes" />

# Codes







    
