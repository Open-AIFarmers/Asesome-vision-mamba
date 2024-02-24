# Paper List
  | Title | Publication&date | summary | recommend |
  | --- | --- | --- | --- |
  |[Deep Multimodal Transfer Learning for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9236655)|TNNLS2022|||
  |[FDDH: Fast Discriminative Discrete Hashing for Large-Scale Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9429177)|TNNLS2022|||
  |[Enhancing Visual Grounding in Vision-Language Pre-training with Position-Guided Text Prompts](https://ieeexplore.ieee.org/document/10363674)|TPAMI2023|改进vlp||
  |[Entity-Graph Enhanced Cross-Modal Pretraining for Instance-Level Product Retrieval](https://ieeexplore.ieee.org/document/10169110)|TPAMI2023|使用了vlp||
  |[Cross-Modal Causal Relational Reasoning for Event-Level Visual Question Answering](https://ieeexplore.ieee.org/document/10146482)|TPAMI2023|视觉问答||
  |[Cross-Modal Retrieval With Partially Mismatched Pairs](https://ieeexplore.ieee.org/document/10050111)|TPAMI2023|有检索但没看出是不是vlp||
  |[Universal Multimodal Representation for Language Understanding](https://ieeexplore.ieee.org/document/10005816)|TPAMI2023|vlp||
  |[Integrating Multi-Label Contrastive Learning With Dual Adversarial Graph Neural Networks for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9815553)|TPAMI2023|虽然是图神经网络做视觉编码器，但可以借鉴一下||  
  |||||
  |[Augment the Pairs: Semantics-Preserving Image-Caption Pair Augmentation for Grounding-Based Vision and Language Models](https://arxiv.org/abs/2311.02536)|WACV2024|数据增强,侧重提高数据集质量，还未发表但已接受，可做借鉴|可参考，不复现|
  |[SeTformer is What You Need for Vision and Language](https://arxiv.org/abs/2401.03540)|AAAI2024|新型Transformer，通过将点积自注意力（DPSA）完全替换为自我最优传输（SeT）来提高性能和计算效率,|可参考，可用来对比|
  |[VL-GPT: A Generative Pre-trained Transformer for Vision and Language Understanding and Generation](https://arxiv.org/abs/2312.09251)|arXiv2023-12-14|不能做baseline，侧重于生成模型，但可以参考，创新点较多，如：图像tokenizer-detokenizer框架、连续视觉嵌入标记（IMG）、提出了更好的预训练方法、还能够理解和生成图像等；感觉和单流模式更相似，需要vision and language transformer model|-|
  |[Prototype-based Aleatoric Uncertainty Quantification for Cross-modal Retrieval](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html)|NeurlPS2023|量化由固有数据模糊性引起的不确定性来提供值得信赖的预测|-|
  |[BCAN: Bidirectional Correct Attention Network for Cross-Modal Retrieval](https://ieeexplore.ieee.org/abstract/document/10138903)|IEEE Transactions on Neural Networks and Learning Systems2023|注意力机制|-|
  |[Cross-Modal Semantic Enhanced Interaction for Image-Sentence Retrieval](https://openaccess.thecvf.com/content/WACV2023/html/Ge_Cross-Modal_Semantic_Enhanced_Interaction_for_Image-Sentence_Retrieval_WACV_2023_paper.html)|WACV2023||-|
  |[AGREE: Aligning Cross-Modal Entities for Image-Text Retrieval Upon Vision-Language Pre-trained Models](https://dl.acm.org/doi/abs/10.1145/3539597.3570481)|WSDM2023|基于vlp模型image-text检索任务的不同模态对齐|-|
  |[Unifying Two-Stream Encoders with Transformers for Cross-Modal Retrieval](https://dl.acm.org/doi/abs/10.1145/3581783.3612427)|MM2023|双流，视觉transformer+语言transformer+对齐|baseline(复现)|
  |[Image as a Foreign Language: BEIT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)|CVPR2023|将ViT和Bert结合，自监督学习|baseline(待定)|
  |[MAMO: Masked Multimodal Modeling for Fine-Grained Vision-Language Representation Learning](https://arxiv.org/abs/2210.04183)|SIGIR2023|联合掩码多模态建模方法|可参考|
  |[Masked Vision and Language Modeling for Multi-modal Representation Learning](https://arxiv.org/abs/2208.02131)|ICLR2023|联合掩蔽视觉和语言建模,而不是独立开发掩蔽语言模型（MLM）和掩蔽图像建模（MIM）|预训练任务可参考|
  |[Universal Vision-Language Dense Retrieval: Learning A Unified Representation Space for Multi-Modal Retrieval ](https://openreview.net/forum?id=PQOlkgsBsik)|ICLR2023|多模态检索统一模型，输入查询，输出多模态数据，与cross modal有区别;pretrained model：vinvl，clip；|-|
  |[BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597)|ACL2023|1、轻量架构QFormer（querying transformer），2、二阶段预训练范式，与LLM模型链接起来；构建图像和文本之间的对齐former，在检索任务上精度较高|baseline|
  |[COTS: Collaborative Two-Stream Vision-Language Pre-Training Model for Cross-Modal Retrieval](https://openaccess.thecvf.com/content/CVPR2022/html/Lu_COTS_Collaborative_Two-Stream_Vision-Language_Pre-Training_Model_for_Cross-Modal_Retrieval_CVPR_2022_paper.html)|CVPR2022|双流VLP模型|可参考，无代码|
  |[TCL: Vision-Language Pre-Training with Triple Contrastive Learning](https://arxiv.org/abs/2202.10401)|CVPR2022|基于ALBEF（通过在多模态特征融合前对齐不同模态之间的表示），ALBEF的对齐是image-text pair之间的全局对齐，没有考虑到模态内部的对齐，以及global-local的细粒度对齐。TCL把ALBEF中的一种对齐方式扩展到了三种对齐方式。|待定|
  |[An Empirical Study of Training End-to-End Vision-and-Language Transformers](https://arxiv.org/abs/2111.02387)|CVPR2022|训练端到端视觉和语言变压器的实证研究，很有意思的一篇文章，测试很多不同部分用不同方法的模型性能，虽然是2022年，但CVPR值得一看|baseline(复现)|
  |[Recall@k Surrogate Loss with Large Batches and Similarity Mixup](https://arxiv.org/abs/2108.11179)|CVPR2022|探索新的损失函数、批量大小、正则化方法之间的相互作用以提高检索的效果|-|
  |[VLDeformer: Vision–Language Decomposed Transformer for fast cross-modal retrieval](https://www.sciencedirect.com/science/article/pii/S0950705122006608)|Knowledge-Based Systems2022|期刊；提升VLtransformer效率，感觉质量高的话可以作为baseline之一。|-|
  |[BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)|ICML2022|boostrapping caption用于“提纯”带噪声web datasets，可参考噪声处理方法|待定|
  |[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334)|ICML2021|a minimal VLP model,体量小速度快，但精度一般，可做baseline|baseline(复现)|
  |[Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision](https://arxiv.org/abs/2102.05918)|ICML2021|ALIGN,十亿噪声数据集|baseline(不复现)|
  |[Align before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)|ICML2021|ALBEF,对比损失，动量蒸馏，混合流模型|baseline|
  |[CLIP:Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)|openai2021||baseline|
  |[VinVL: Revisiting Visual Representations in Vision-Language Models](https://arxiv.org/abs/2101.00529)|CVPR2021||baseline|
  |||||
  |[Student Can Also be a Good Teacher: Extracting Knowledge from Vision-and-Language Model for Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3459637.3482194)|CIKM2021|VLP based on transformer||
  |[Fine-Grained Cross-Modal Retrieval with Triple-Streamed Memory Fusion Transformer Encoder](https://ieeexplore.ieee.org/document/9859738)|ICME2022|这篇是为了解决transformer的效率和有效性之间的矛盾，数据集和思路有借鉴意义，如果出名可做baseline之一||
  |[FaD-VLP: Fashion Vision-and-Language Pre-training towards Unified Retrieval and Captioning](https://aclanthology.org/2022.emnlp-main.716/)|EMNLP2022|时尚领域VLP范式||
  |[Image Retrieval on Real-Life Images With Pre-Trained Vision-and-Language Models](https://openaccess.thecvf.com//content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html)|ICCV2021|composed image retrieval：输入为“图像+修改文字”，目标输出符合输入的图像；VLP范式，可能具有一定参考意义||
  |[TVT: Three-Way Vision Transformer through Multi-Modal Hypersphere Learning for Zero-Shot Sketch-Based Image Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/20136)|AAAI2022|草图图像检索，任务有偏差，参考价值可能不大||
  |[Vision Transformer Hashing for Image Retrieval](https://ieeexplore.ieee.org/document/9859900)|ICME2022|数据集： CIFAR10、ImageNet、NUS-Wide 和 COCO；单模态||
  |[Bidirectional Focused Semantic Alignment Attention Network for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9414382)|ICASSP2021|双向注意力机制消除无关语义信息影响，可能具有借鉴意义||
  | [Bit-aware Semantic Transformer Hashing for Multi-modal Retrieval](https://dl.acm.org/doi/10.1145/3477495.3531947) | SIGIR2022 |数据集：MIR Flickr, NUS-WIDE and MS COCO；transformer+hashing||
  |[Learning Homogeneous and Heterogeneous Co-Occurrences for Unsupervised Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9428240)|ICME2021|无transformer，但数据集上有借鉴意义||
  |[Contrastive Label Correlation Enhanced Unified Hashing Encoder for Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3511808.3557265)| CIKM2022 |transformer+hashing||
  | [Joint Feature Synthesis and Embedding: Adversarial Cross-Modal Retrieval Revisited](https://ieeexplore.ieee.org/document/9296975) | TPAMI2022 |GAN||
  | [Universal Weighting Metric Learning for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9454290) | TPAMI2022 |跨模态检索通用的加权度量学习框架||
  |[Token Embeddings Alignment for Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3503161.3548107)|MM2022|词嵌入传入multi-modal transformer中，但侧重的是词嵌入对齐的创新，借鉴意义不大|| 
  |||||
  | [MTFH: A Matrix Tri-Factorization Hashing Framework for Efficient Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/8827941) | TPAMI2021 |hashing|-|
  |[Bi-CMR: Bidirectional Reinforcement Guided Hashing for Effective Cross-Modal Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/21268)| AAAI2022 |双向引导哈希|-|
  |[Dual Adversarial Graph Neural Networks for Multi-label Cross-modal Retrieval](https://ojs.aaai.org/index.php/AAAI/article/view/16345)| AAAI2021 |对抗+图神经网络|-|
  |[Efficient Cross-Modal Retrievalvia Deep Binary Hashing and Quantization](https://www.bmvc2021-virtualconference.com/assets/papers/1202.pdf)| BMVC2021 |hashing|-|
  |[OTCMR: Bridging Heterogeneity Gap with Optimal Transport for Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3459637.3482158)|CIKM2021|CNN|-|
  |[Cross Modal Retrieval With Querybank Normalisation](https://openaccess.thecvf.com//content/CVPR2022/html/Bogolin_Cross_Modal_Retrieval_With_Querybank_Normalisation_CVPR_2022_paper.html)|CVPR2022|text-video|-|
  |[ViSTA: Vision and Scene Text Aggregation for Cross-Modal Retrieval](https://openaccess.thecvf.com//content/CVPR2022/html/Cheng_ViSTA_Vision_and_Scene_Text_Aggregation_for_Cross-Modal_Retrieval_CVPR_2022_paper.html)|CVPR2022|场景文本|-|
  |[Cross-Modal Center Loss for 3D Cross-Modal Retrieval](https://openaccess.thecvf.com/content/CVPR2021/html/Jing_Cross-Modal_Center_Loss_for_3D_Cross-Modal_Retrieval_CVPR_2021_paper.html)|CVPR2022|三维数据(点云、mesh等)|-|
  |[Learning Cross-Modal Retrieval With Noisy Labels](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_Learning_Cross-Modal_Retrieval_With_Noisy_Labels_CVPR_2021_paper.html)|CVPR2021|鲁棒性；image：CNN|-|
  |[Probabilistic Embeddings for Cross-Modal Retrieval](https://openaccess.thecvf.com/content/CVPR2021/html/Chun_Probabilistic_Embeddings_for_Cross-Modal_Retrieval_CVPR_2021_paper.html)|CVPR2021|不同模态样本表示为共同嵌入空间的概率分布|-|
  |[A General Framework For Incomplete Cross-Modal Retrieval With Missing Labels And Missing Modalities](https://ieeexplore.ieee.org/document/9747813)|ICASSP2022|最大化未缺失标签的数据对之间的相关性；CNN|-|
  |[Deep Adversarial Quantization Network for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9414247)|ICASSP2021|对抗学习对齐不同模态；无transformer|-|
  |[Scalable Discriminative Discrete Hashing For Large-Scale Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9413871)|ICASSP2021|hashing；无transformer|-|
  |[Ask&Confirm: Active Detail Enriching for Cross-Modal Retrieval With Partial Query](https://openaccess.thecvf.com//content/ICCV2021/html/Cai_AskConfirm_Active_Detail_Enriching_for_Cross-Modal_Retrieval_With_Partial_Query_ICCV_2021_paper.html)|ICCV2021|以 "询问-确认"的方式进行交互过程，主动搜索当前查询中缺少的辨别细节|-|
  |[Wasserstein Coupled Graph Learning for Cross-Modal Retrieval](https://openaccess.thecvf.com//content/ICCV2021/html/Wang_Wasserstein_Coupled_Graph_Learning_for_Cross-Modal_Retrieval_ICCV_2021_paper.html)|ICCV2021|无transformer|-|
  |[A Channel Mix Method for Fine-Grained Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9859609)|ICME2022|无transformer|-|
  |[Instance-Level Semantic Alignment for Zero-Shot Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9860026)|ICME2022|无transformer|-|
  |[Learning Controlled Semantic Embedding for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9428280)|ICME2021|无transformer|-|
  |[Efficient Online Label Consistent Hashing for Large-Scale Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9428323)|ICME2021|无transformer|-|
  |[Attention-Guided Semantic Hashing for Unsupervised Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9428330)|ICME2021|attention注意力机制|-|
  |[Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective](https://www.ijcai.org/proceedings/2021/454)|IJCAI2021|无transformer|-|
  |[A Fast Discrete Two-Step Learning Hashing for Scalable Cross-Modal Retrieval](https://www.isca-archive.org/interspeech_2021/zhao21b_interspeech.html)|INTERSPEECH2021|无transformer|-|
  |[Early-Learning regularized Contrastive Learning for Cross-Modal Retrieval with Noisy Labels](https://dl.acm.org/doi/10.1145/3503161.3548066)|MM2022|无transformer|-|
  |[Cross-Modal Retrieval with Heterogeneous Graph Embedding](https://dl.acm.org/doi/10.1145/3503161.3548195)|MM2022|无transformer|-|
  |[Deep Evidential Learning with Noisy Correspondence for Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3503161.3547922)|MM2022|无transformer|-|
  |[Joint-teaching: Learning to Refine Knowledge for Resource-constrained Unsupervised Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475286)|MM2021|无transformer|-|
  |[Exploring Graph-Structured Semantics for Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475567)|MM2021|无transformer|-|
  |[A Differentiable Semantic Metric Approximation in Probabilistic Embedding for Cross-Modal Retrieval](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4e786a87e7ae249de2b1aeaf5d8fde82-Abstract-Conference.html)|NIPS2022|无transformer|-|
  |[Multimodal Disentanglement Variational AutoEncoders for Zero-Shot Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3477495.3532028)|SIGIR2022|无transformer|-|
  |[PAN: Prototype-based Adaptive Network for Robust Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3404835.3462867)|SIGIR2021|无transformer|-|
  |[Heterogeneous Attention Network for Effective and Efficient Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3404835.3462924)|SIGIR2021|无transformer|-|
  |[FedCMR: Federated Cross-Modal Retrieval](https://dl.acm.org/doi/10.1145/3404835.3462989#sec-comments)|SIGIR2021|无transformer|-|
  |[Joint Specifics and Consistency Hash Learning for Large-Scale Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9850431/metrics#metrics)|TIP2022|无transformer|-|
  |[Deep Relation Embedding for Cross-Modal Retrieval ](https://ieeexplore.ieee.org/document/9269483)|TIP2021|无transformer|-|
  |[Asymmetric Supervised Consistent and Specific Hashing for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9269445)|TIP2021|无transformer|-|
  |[Cross-Domain Image Captioning via Cross-Modal Retrieval and Model Adaptation](https://ieeexplore.ieee.org/document/9292444)|TIP2021|无transformer|-|
  |[Deep Multimodal Transfer Learning for Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9236655)|TNNLS2022|无transformer|-|
  |[FDDH: Fast Discriminative Discrete Hashing for Large-Scale Cross-Modal Retrieval](https://ieeexplore.ieee.org/document/9429177)|TNNLS2022|无transformer|-|
  |[Cross-Modal Retrieval Augmentation for Multi-Modal Classification](https://aclanthology.org/2021.findings-emnlp.11/)|EMNLP2021|retrieval augmentation，无transformer|-|
  |[Unsupervised Contrastive Hashing for Cross-Modal Retrieval in Remote Sensing](https://ieeexplore.ieee.org/document/9746251)|ICASSP2022|遥感领域，无transformer|-|
  |[Learning Disentangled Factors from Paired Data in Cross-Modal Retrieval: An Implicit Identifiable VAE Approach](https://dl.acm.org/doi/10.1145/3474085.3475448)|MM2021|无transformer|-|
  |[MCCN: Multimodal Coordinated Clustering Network for Large-Scale Cross-modal Retrieval](https://dl.acm.org/doi/10.1145/3474085.3475670)|MM2021|无transformer|-|
  |[Learning Text-image Joint Embedding for Efficient Cross-modal Retrieval with Deep Feature Engineering](https://dl.acm.org/doi/10.1145/3490519)|TOIS2022|无transformer|-|
  |[StacMR: Scene-Text Aware Cross-Modal Retrieval](https://openaccess.thecvf.com//content/WACV2021/html/Mafla_StacMR_Scene-Text_Aware_Cross-Modal_Retrieval_WACV_2021_paper.html)|WACV2021|无transformer|-|
  
  
  
# 数据集
![数据集](https://cdn.nlark.com/yuque/0/2023/png/25419362/1687616552109-46cc65ca-d84b-4d70-b6c5-5c07acac4f00.png#averageHue=%23f7f5f4&clientId=uce2552bf-d2ca-4&from=paste&height=438&id=u20e72a73&originHeight=547&originWidth=1317&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=56362&status=done&style=none&taskId=uf65a463e-5eb9-462e-b39c-494757e84e5&title=&width=1053.6)

# Baseline

# 模型解构
## 主流多模态检索技术路线
![ca1d9625fb9da876a5353df9ef186cb](https://github.com/lpf992/vision-mamba/assets/151422800/82136247-9e56-4218-aaa4-d978a3356a84)
* 单流
* 双流*
