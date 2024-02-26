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
|title|publication&date|summary|data&cost|recommendation|
|---|---|---|---|---|
|[ESA: External Space Attention Aggregation for Image-Text Retrieval](https://ieeexplore.ieee.org/document/10061465) |tcsvt2023|[**code**](https://github.com/KevinLight831/ESA);网络架构cnn+bert+esa|Flicker30K：1000 images for validation, 1000 images for testing, and 29000 images for training；<br> MS-COCO：113,287 training images, 5000 test images, and 5000 validation images；<br> 所有的训练都在一张**RTX 3090**|baseline(复现)|
|[X2-VLM: All-In-One Pre-trained Model For Vision-Language Tasks](https://arxiv.org/abs/2211.12402)|tpami2023|[**code**](github.com/zengyan-97/X2-VLM);有可**下载参数**;统一框架预训练的一体化VLM，用于多粒度视觉语言对齐。模块化：X2-VLM具有灵活的模块化架构，分别具有视觉、文本和融合三个模块。所有模块都基于transformer|**预训练数据**：4M dataset[COCO and Visual Genome (VG), SBU Captions and Conceptual Captions (CC)],include annotations for COCO and VG images from RefCOCO, GQA, and Flickr entities following OFA and MDETR. **scale up the pre-training dataset** by including out-of-domain and much noisier image-text pairs from Conceptual 12M dataset (CC-12M) and LAION, and object annotations from Objects365 and OpenImages;<br>**微调**：most downstream tasks are built on top of COCO and VG；<br>**多模态检索任务评估**：MSCOCO,Flickr30K；<br>![image](https://github.com/Open-AIFarmers/Asesome-vision-mamba/assets/65701521/d704b161-c3b5-4b2d-8152-34f8acaf832b)|baseline(复现)|
|[Enhancing Visual Grounding in Vision-Language Pre-training with Position-Guided Text Prompts](https://ieeexplore.ieee.org/document/10363674)|TPAMI2023|[**code**](https://github.com/sail-sg/ptp);有**可下载参数**；cvpr转期刊；改进vlp，位置引导文本提示，增强vlp的视觉基础能力|**预训练数据**：in earlier studies，4M dataset[COCO,VG,SBU and CC3M];14M setting is a combination of 4M setting and CC-12M；**large**：DataComp-1B(1.17B), MMC4(324M), and LAION400M(375.3M><br>**图像文本检索**：evaluate on MSCOCO,FLICKR30K<br>**cost**:预训练需要8张a100|baseline(复现)|
|[Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)|CVPR2023|[**code**](https://github.com/microsoft/unilm/tree/master/beit3);有可**下载参数**;将ViT和Bert结合，自监督学习；|**预训练数据**：多模态：Conceptual 12M (CC12M)、Conceptual Captions (CC3M)、SBU Captions (SBU)、COCO、Visual Genome (VG)；单模态：14M images from ImageNet-21K and 160GB text corpora from English Wikipedia, BookCorpus, OpenWebText3, CC-News, and Stories.<br>**微调数据集**：COCO and Flickr30K|baseline(复现)|
|[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|ACL2023|[**code**](https://github.com/salesforce/LAVIS/tree/main/projects/blip2);有可**下载参数**;1、轻量架构QFormer（querying transformer），2、二阶段预训练范式，与LLM模型链接起来；构建图像和文本之间的对齐former，在检索任务上精度较高|**预训练数据**：COCO、Visual Genome、CC3M、CC12M、SBU and 115M images from the LAION400M dataset;<br>**预训练cost**：using a single 16-A100(40G) machine, our largest model with ViT-g and FlanT5-XXL requires less than 6 days for the first stage and less than 3 days for the second stage.<br>**微调数据集**：COCO and Flickr30K,图像文本检索只需要微调第一阶段|baseline(复现)|
|[An Empirical Study of Training End-to-End Vision-and-Language Transformers](https://arxiv.org/abs/2111.02387)|CVPR2022|[**code**](https://github.com/zdou0830/METER);有可**下载参数**;训练端到端视觉和语言变压器的实证研究，很有意思的一篇文章，测试很多不同部分用不同方法的模型性能，虽然是2022年，但CVPR值得一看|**预训练参数**：COCO, Conceptual Captions, SBU Captions, and Visual Genome.<br>**微调数据集**：COCO、Flickr30k；<br>**预训练cost**：8个A100，METER-CLIP-ViTBASE−32：3天；METER-SwinBASE and METER-CLIP-ViTBASE−16：8天|baseline(复现)|
|[Fast, Accurate, and Lightweight Memory-Enhanced Embedding Learning Framework for Image-Text Retrieval](https://ieeexplore.ieee.org/document/10414133)|tcsvt2024|没有用vlp预训练解构，不同模态独立嵌入，特征提取+跨模态图网络+记忆网络;这篇属于偏传统的方法，但整体思路挺有意思，但好像没代码|**评估数据集**：Flickr30K and MS-COCO, <br>**cost**评估速度用了一个RTX3090|baseline要代码|
||||||
||||||
||||||
|[Semantic Pre-alignment and Ranking Learning with Unified Framework for Cross-modal Retrieval](https://ieeexplore.ieee.org/document/9794649)|tcsvt2022|由三个子网络组成：视觉网络、文本网络和交互网络（对齐）。网络结构可参考，但未使用pretrained架构，无transformer，未涉及vlp模型;无代码暂不考虑|||
|[Image-Text Retrieval With Cross-Modal Semantic Importance Consistency](https://ieeexplore.ieee.org/document/9940913)|tcsvt2022|单流;无代码|||
|[Relation-Aggregated Cross-Graph Correlation Learning for Fine-Grained Image–Text Retrieval](https://ieeexplore.ieee.org/document/9829420)|tnnls2022|有图像文本检索，但视觉用的是CNN，不知道可不可以做baseline||baseline(无代码不复现)|
|[Deep Multimodal Transfer Learning for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9236655)|TNNLS2022|深度多模态迁移学习||baseline(不复现)|
|[Cross-Modal Retrieval With Partially Mismatched Pairs](https://ieeexplore.ieee.org/document/10050111)|TPAMI2023|不是vlp，提出了新的框架，但没有用图像文本检索|||
|[Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval](https://dl.acm.org/doi/abs/10.1145/3581783.3612427)|MM2023|双流，视觉transformer+语言transformer+对齐|||
|[COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_COTS_Collaborative_Two-Stream_Vision-Language_Pre-Training_Model_for_Cross-Modal_Retrieval_CVPR_2022_paper.html)|CVPR2022|双流VLP模型||可参考，无代码|
|[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)|ICML2021|a minimal VLP model,体量小速度快，但精度一般，可做baseline|[Microsoft COCO (MSCOCO)、Visual Genome (VG)、SBU Captions (SBU)、Google Conceptual Captions (GCC)] [pre-train ViLT-B/32 for 100K or 200K steps on 64 NVIDIA V100 GPUs with a batch size of 4,096]|baseline(不复现)|
|[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)|ICML2021|ALIGN,十亿噪声数据集||baseline(不复现)|
|[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)|ICML2021|ALBEF,对比损失，动量蒸馏，混合流模型||baseline(不复现)|
|[CLIP:Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)|openai2021|||baseline(不复现)|
|[VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529)|CVPR2021|||baseline(不复现)|
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

1. **ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision** *Wonjae Kim, Bokyung Son, Ildoo Kim* 2021 [paper] [baseline(复现)] (https://arxiv.org/abs/2102.03334)
2. **Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision** *Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, Tom Duerig* 2021 [paper] [baseline(不复现)] (https://arxiv.org/abs/2102.05918)
3. **Align before Fuse: Vision and Language Representation Learning with Momentum Distillation** ** 2021 [paper] [baseline] (https://arxiv.org/abs/2107.07651)
4. **CLIP:Learning Transferable Visual Models From Natural Language Supervision** *Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever* [paper] [baseline] (https://arxiv.org/abs/2103.00020)|openai2021||baseline|
5. **VinVL: Revisiting Visual Representations in Vision-Language Models** *Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, Jianfeng Gao* 2021 [paper] [baseline] (https://arxiv.org/abs/2101.00529)
6. **Student Can Also be a Good Teacher: Extracting Knowledge from Vision-and-Language Model for Cross-Modal Retrieval** *Jun Rao,Tao Qian,Shuhan Qi,Yulin Wu,Qing Liao,Xuan Wang* 2021 [paper] (https://dl.acm.org/doi/10.1145/3459637.3482194)
7. **Image Retrieval on Real-Life Images With Pre-Trained Vision-and-Language Models** *Zheyuan Liu, Cristian Rodriguez-Opazo, Damien Teney, Stephen Gould* 2021 [paper] [可参考] (https://openaccess.thecvf.com//content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)
8. **Learning Homogeneous and Heterogeneous Co-Occurrences for Unsupervised Cross-Modal Retrieval** ** 2021 [paper] [数据集有借鉴意义] (https://ieeexplore.ieee.org/document/9428240)
9. 
  
<a name="2022" />

### 2.2 2022

1. **COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval** *Haoyu Lu, Nanyi Fei, Yuqi Huo, Yizhao Gao, Zhiwu Lu, Ji-Rong Wen* 2022 [paper] [可参考，无代码] (https://openaccess.thecvf.com/content/CVPR2022/html/Lu_COTS_Collaborative_Two-Stream_Vision-Language_Pre-Training_Model_for_Cross-Modal_Retrieval_CVPR_2022_paper.html)
2. **TCL: Vision-Language Pre-Training with Triple Contrastive Learning** *Jinyu Yang, Jiali Duan, Son Tran, Yi Xu, Sampath Chanda, Liqun Chen, Belinda Zeng, Trishul Chilimbi, Junzhou Huang* 2022 [paper] [待定] (https://arxiv.org/abs/2202.10401)
3. **An Empirical Study of Training End-to-End Vision-and-Language Transformers** *Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, Zicheng Liu, Michael Zeng* 2022 [paper] [baseline](https://arxiv.org/abs/2111.02387)
4. **BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation** *Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi* 2022 [paper] [待定] (https://arxiv.org/abs/2201.12086)
5. **Fine-Grained Cross-Modal Retrieval with Triple-Streamed Memory Fusion Transformer Encoder** ** 2022 [paper] (https://ieeexplore.ieee.org/document/9859738)
6. **FaD-VLP: Fashion Vision-and-Language Pre-training towards Unified Retrieval and Captioning** *Suvir Mirchandani, Licheng Yu, Mengjiao Wang, Animesh Sinha, Wenwen Jiang, Tao Xiang, Ning Zhang* 2022 [paper] (https://aclanthology.org/2022.emnlp-main.716/)
7. **Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval** *Hongfa Wu,Lisai Zhang,Qingcai Chen,Yimeng Deng,Joanna Siebert,Yunpeng Han,Zhonghua Li,Dejiang Kong,Zhao Cao* 2022 [paper] (https://dl.acm.org/doi/10.1145/3511808.3557265)
8. **Joint Feature Synthesis and Embedding: Adversarial Cross-Modal Retrieval Revisited** *Xing Xu, Kaiyi Lin, Yang Yang, Alan Hanjalic, Heng Tao Shen* 2022 [paper] (https://ieeexplore.ieee.org/document/9296975)
9. **Universal Weighting Metric Learning for Cross-Modal Retrieval** *Jiwei Wei, Xing Xu, Yang Yang, Yanli Ji, Zheng Wang, Heng Tao Shen* 2022 [paper] (https://ieeexplore.ieee.org/document/9454290)
10. **Token Embeddings Alignment for Cross-Modal Retrieval** *Chen-Wei Xie,Jianmin Wu,Yun Zheng,Pan Pan,Xian-Sheng Hua* 2022 [paper](https://dl.acm.org/doi/10.1145/3503161.3548107)
11. **Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval** *Wentao Tan,Lei Zhu,Weili Guan,Jingjing Li,Zhiyong Cheng* 2022 [paper](https://dl.acm.org/doi/10.1145/3477495.3531947)
12. **TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval** *Jialin Tian,Xing Xu,Fumin Shen,Yang Yang,Heng Tao Shen* 2022 [paper] (https://ojs.aaai.org/index.php/AAAI/article/view/20136)
13. **Vision Transformer Hashing for Image Retrieval** *Shiv Ram Dubey, Satish Kumar Singh, Wei-Ta Chu* [paper] (https://ieeexplore.ieee.org/document/9859900)
14. 

<a name="2023" />

### 2.3 2023

1. **Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval** *Yi Bin, Haoxuan Li, Yahui Xu, Xing Xu, Yang Yang, Heng Tao Shen* 2023 [paper] [baseline] (https://dl.acm.org/doi/abs/10.1145/3581783.3612427)
2. **Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks** *Wenhui Wang, Hangbo Bao, Li Dong, Johan Bjorck, Zhiliang Peng, Qiang Liu, Kriti Aggarwal, Owais Khan Mohammed, Saksham Singhal, Subhojit Som, Furu Wei* 2023 [paper] [baseline] (https://arxiv.org/abs/2208.10442)
3. **MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning** *Zijia Zhao, Longteng Guo, Xingjian He, Shuai Shao, Zehuan Yuan, Jing Liu* 2023 [paper] [可参考] (https://arxiv.org/abs/2210.04183)
4. **Masked Vision and Language Modeling for Multi-modal Representation Learning** *Gukyeong Kwon, Zhaowei Cai, Avinash Ravichandran, Erhan Bas, Rahul Bhotika, Stefano Soatto* 2023 [paper] [可参考] (https://arxiv.org/abs/2208.02131)
5. **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models** *Junnan Li, Dongxu Li, Silvio Savarese, Steven Hoi* 2023 [paper] [baseline] (https://arxiv.org/abs/2301.12597)
6. 




<a name="2024" />

### 2.4 2024

1. **Augment the Pairs: Semantics-Preserving Image-Caption Pair Augmentation for Grounding-Based Vision and Language Models.** *Jingru Yi, Burak Uzkent, Oana Ignat, Zili Li, Amanmeet Garg, Xiang Yu, Linda Liu* 2024 [paper] (https://arxiv.org/abs/2311.02536)
2. **SeTformer is What You Need for Vision and Language.** *Pourya Shamsolmoali, Masoumeh Zareapoor, Eric Granger, Michael Felsberg* 2024 [paper] [vision and language transformer] (https://arxiv.org/abs/2401.03540)
3. 


<a name="model" />

# 模型解构

## 主流多模态检索技术路线
![ca1d9625fb9da876a5353df9ef186cb](https://github.com/lpf992/vision-mamba/assets/151422800/82136247-9e56-4218-aaa4-d978a3356a84)
- 单流
- 双流*

<a name="Codes" />

# Codes







    
