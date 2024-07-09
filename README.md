# Awesome Efficient Diffusion

[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

> A curated list of methods that focus on improving the efficiency of diffusion models

### Updates

> I‘m trying to update this list weekly (every monday morning) from my personal knowledge stack, and collect each conference's proceedings. If you find this repo useful, it would be kind to consider **★staring** it or **☛contributing to** it. 

- [2024/07/08] Reorganizing the catalogs
- [2024/07/09] (ING) Filling in existing surveys 

### Catalogs

- [0. Basics](#basics)
    - [0.1 Diffusion Formulations](#diffusion-formulation)
    - [0.2 Solvers](#solvers)
    - [0.3 Models](#models)
        - [0.3.1 Key Components](#key-components)
        - [0.3.2 Open-sourced Models](#open-sourced-models)
        - [0.3.3 Closed-source Models/Products](#closed-source-models)
    - [0.4 Datasets for Train/Eval](#datasets)
        - [0.4.1 Unconditional](#unconditional)
        - [0.4.2 Unconditional](#class-conditioned)
        - [0.4.3 Text-to-Image](#text-to-image)
    - [0.5 Evaluation Metrics](#evaluation-metrics)
    - [0.6 Miscellaneous](#miscellaneous)
        - [0.6.1 Video Generation](#video-generation)
        - [0.6.2 Customized Generation](#customized-generation)
        - [0.6.2 Generating Complex Scene](#generate-complex-scene)

- [1. Algorithm-level](#algorithms)
    - [1.1 Timestep Reduction](#timestep-reduction)
        - [1.1.1 Efficient Solver](#efficient-solver)
        - [1.1.2 Timestep Distillation](#timestep-distillation)
    - [1.2 Model Architecture (Weight Reduction)](#architecture-level-compression)
        - [1.2.1 Prune/Distillation](#pruning)
        - [1.2.2 Adaptive Architecture](#adaptive-architecture)
    - [1.3 Token-level Compression](#token-level-compression)
        - [1.3.1 Token Reduction](#token-reduction)
        - [1.3.2 Patched Inference](#patched-inference)
    - [1.4 Model Quantization](#model-quantization)
    - [1.5 Efficient Tuning](#efficient-tuning)
         - [1.5.1 Low Rank](#token-reduction)

- [2. System-level](#systems)
    - [2.1 GPU](#gpu)
    - [2.2 Mobile](#mobile)

# Basics


### Resources

> Recommended introductory learning materials

- [David Saxton's Tutorial on Diffusion](https://saxton.ai/diffusion/00_index.html)
- [Song Yang's Post: Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [EfficientML Course, MIT, Han Song, The Diffusion Chapter](https://www.dropbox.com/scl/fi/q8y9ap7mlucmiimyh3zl5/lec16.pdf?rlkey=6wx4z3pnhic8pq0oju8ro3qzr&e=1&dl=0)

---

- [Applications: Huggingface Diffuser Doc](https://huggingface.co/docs/diffusers/index)

## Diffusion Formulation

> formulations of diffusion, development of theory

- [**DPM**] "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"; 
    - Early advance of diffusion formulation
    - 2015/03 | ICML15 | [[Paper](https://arxiv.org/abs/1503.03585)]


- [**DDPM**]  "Denoising Diffusion Probabilistic Models"; 
    - 2020/06 | NeurIPS20 | [[Paper](https://arxiv.org/abs/2006.11239)]
    - The discrete time diffusion

- [**SDE-based Diffusion**]
    - 2020/11 | ICLR21 | [[Paper](https://arxiv.org/abs/2011.13456)]
    - Continuous time Neural SDE formulation of diffusion

---

> how to introduce control signal

- [**Classifier-based Guidance**] "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"; 
    - 2021/05 | Arxiv2105 | [[Paper](https://arxiv.org/abs/2105.05233)]
    - Introduce control signal through classifier

- [**Classifier-free Guidance (CFG)**] "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"; 
    - 2022/07 | NeurIPS 2021 Workshop | [[Paper](https://arxiv.org/abs/2207.12598)]
    - Introduce CFG, jointly train a conditional and an unconditional diffusion model, and combine them

- [**LDM**] "High-Resolution Image Synthesis with Latent Diffusion Models";
    - 2021/12 | CVPR22 | [[Paper](https://arxiv.org/abs/2112.10752)] | [[Code](https://github.com/CompVis/stable-diffusion)]
    - Text-to-image conditioning with cross attention
    - Latent space diffusion model


## Solvers

- [**DDIM**]: "Denoising Diffusion Implicit Models";
    - 2020/10 | ICLR21 | [[Paper](https://arxiv.org/abs/2010.02502)]
    - determinstic sampling, skip timesteps

- [**DPMSolver**]: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps";
    - 2022/06 | NeurIPS22 | [[Paper](https://arxiv.org/abs/2206.00927)]
    - utilize the sub-linear property of ODE solving, converge in 10-20 steps 
    
- [**DPMSolver++**]: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models";
    - 2022/11 | Arxiv | [[Paper](https://arxiv.org/abs/2211.01095)]
    - multi-order ODE, faster convergence

## Models

### Key Components

> Text_encoder

- [**CLIP**] "Learning Transferable Visual Models From Natural Language Supervision";
    - 2021/03 | Arxiv | [[Paper](https://arxiv.org/abs/2103.00020)]
    - *Containing Operations:*
        - Self-Attention (Cross-Attention)
        - FFN (FC)
        - LayerNorm (GroupNorm)]

- [**T5**] "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer";
    - 2019/10 | Arxiv | [[Paper](https://arxiv.org/abs/1910.10683)]
    - *Containing Operations:*
        - Self-Attention (Cross-Attention)
        - FFN (FC)
        - LayerNorm (GroupNorm)

Summarization of adopted text encoders for large text-to-image models from [Kling-AI Technical Report](https://github.com/Kwai-Kolors/Kolors/blob/master/imgs/Kolors_paper.pdf)

![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20240709140445.png)



> VAE (for latent-space)

- [**VAE**] "Tutorial on Variational Autoencoders";
    - 2016/06 | Arxiv | [[Paper](https://arxiv.org/abs/1606.05908)]
    - *Containing Operations:*
        - Conv
        - DeConv (ConvTransposed, Interpolation)

> Diffusion Network

- [**U-Net**] "U-Net: Convolutional Networks for Biomedical Image Segmentation";
    - 2015/05 | Arxiv | [[Paper](https://arxiv.org/abs/1505.04597)]
    - *Containing Operations:*
        - Conv
        - DeConv (ConvTransposed, Interpolation)
        - Low-range Shortcut Connection

- DiT

> UpScaler



### Open-sourced Models


- [**Imagen**]: "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding”;
    - 2022/05 | NeurIPS22 | [[Paper](https://arxiv.org/abs/2205.11487)]

- [**DeepFlyoid-IF**] "DeepFlyod-IF";
    - 2022/04 | Arxiv | Stability.AI | [[Technical Report](https://github.com/deep-floyd/IF)] | [[Code](https://github.com/deep-floyd/IF)]
    - Larger Language Model (T5 over CLIP) | Pixel-space Diffusion | Diffusion for SR

### Closed-source Models





---



## Datasets

### Unconditional


### Class-Conditioned

- **CIFAR-10**: 

- **CelebA**:

### Text-to-image

## Evaluation Metrics

- [InceptionDistance](https://huggingface.co/docs/diffusers/v0.16.0/en/conceptual/evaluation#quantitative-evaluation)
    - **Fréchet Inception Distance**: evaluting 2 **set** of image, intermediate feature distance of InceptionNet between reference image and generated image, lower the better
    - **Kernel Inception Distance**
    - **Inception Score**
    - *limitation*: when model trained under large image-caption dataset ([LAION-5B](https://laion.ai/blog/laion-5b/)), for that the Inception is pre-trained on ImageNet-1K. (StableDiffusion pre-trained set may have overlap)
        - The specific Inception model used during computation.
        - The image format (not the same if we start from PNGs vs JPGs)

- [Clip-related](https://huggingface.co/docs/diffusers/v0.16.0/en/conceptual/evaluation#quantitative-evaluation)
    - **CLIP score**: compatibility of image-text pair
    - **CLIP directional similarity**: compatibility of image-text pair
    - *limitation*: The captions tags were crawled from the web, may not align with human description.

- [Other Metrics (Refering from Schuture/Benchmarking-Awesome-Diffusion-Models)](https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models)

## Miscellaneous

### Video Generation


### Customized Generation


### Generate Complex Scene


# Algorithm-level

## Timestep Reduction

> reduce the timestep (the number of u-net inference)


### Efficient Solver

- [**DDIM**]: "Denoising Diffusion Implicit Models";
    - 2021/10 | ICLR21 | [[Paper](https://arxiv.org/abs/2010.02502)]
    - **📊 Key results**: 50~100 Steps -> 10~20 Steps with moderate performance loss

- [**DPM-Solver**]: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps";
    - 2022/06 | NeurIPS | [[Paper](https://arxiv.org/abs/2206.00927)]
    - **📊 Key results**: NFE (number of U-Net forward) = 10 achieves similar performance with DDIM NFE = 100
### Timestep Distillation

- [**Catch-Up Distillation**]: "Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling";
    - 2023/05 | Arxiv2305 | [[Paper](https://arxiv.org/abs/2305.10769)]

- [**ReDi**]: "ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval";
    - Skip intermediate steps:
    - Retrieval: find similar partially generated scheduling in early stage
    - 2023/02 | ICML23 | [[Paper](https://arxiv.org/abs/2302.02285)]

- [**Consistency Model**]: "Consistency Models";
    - New objective: consistency based
    - 2023/03 | Arxiv2303 | [[Paper](https://arxiv.org/pdf/2303.01469.pdf)]]

## Architecture-level Compression

> reduce the diffusion model cost (the repeatedly inference u-net) with  *pruning* / *neural architecture search (nas)* techniques

### Pruning

- [**Structural Pruning**]: "Structural Pruning for Diffusion Models";
    - 2023/05 | Arxiv2305 | [[Paper](https://arxiv.org/abs/2305.10924)] [[Code](https://github.com/VainF/Diff-Pruning)]

### Adaptive Architecture

> adaptive skip part of the architecture across timesteps

## Token-level Compression  

### Token Reduction

> save computation for different sample condition (noise/prompt/task)

- [**ToMe**]: "Token Merging for Fast Stable Diffusion";
    - 2023/03 | Arxiv2304 | [[Paper](https://arxiv.org/abs/2303.17604)] [[Code](https://github.com/dbolya/tomesd)]

### Patched Inference

> reduce the processing resolution

- [**PatchDiffusion**]: "Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models";
    - 2023/04 | NeurIPS23 | [[Paper](https://arxiv.org/abs/2304.12526)]

- [**MemEffPatchGen**]: "Memory Efficient Diffusion Probabilistic Models via Patch-based Generation";
    - 2023/04 | CVPR23W | [[Paper](https://arxiv.org/abs/2304.07087)]  
## Model Quantization

> quantization & low-bit inference/training

- [**PTQD**]: "PTQD: Accurate Post-Training Quantization for Diffusion Models";
    - 2023/05 | NeurIPS23 | [[Paper](https://arxiv.org/abs/2305.10657)]

- [**BiDiffusion**]: "Binary Latent Diffusion";
    - 2023/04 | Arxiv2304 | [[Paper](https://arxiv.org/abs/2304.04820)]

## Efficient Tuning

- [**DiffFit**]: "DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning";
    - 2023/04 | Arxiv2304 | [[Paper](https://arxiv.org/abs/2304.06648)]

- [**ParamEffTuningSummary**]: "A Closer Look at Parameter-Efficient Tuning in Diffusion Models";
    - 2023/03 | Arxiv2303 | [[Paper](https://arxiv.org/abs/2303.18181)]

### 5.1. Low-Rank 

> The LORA family




# System-level

## GPU

## Mobile

- [**SnapFusion**]: "SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds";
    - Platform: iPhone 14 Pro, 1.84s
    - Model Evolution: 3.8x fewer parameters compared to SD-V1.5
    - Step Distillation into 8 steps
    - 2023/06 | Arxiv2306 | [[Paper](http://arxiv.org/abs/2306.00980)]

---

# Related Resources

- [heejkoo/Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [awesome-stable-diffusion/awesome-stable-diffusion](https://github.com/awesome-stable-diffusion/awesome-stable-diffusion)
- [hua1995116/awesome-ai-painting](https://github.com/hua1995116/awesome-ai-painting)
- [PRIV-Creation/Awesome-Diffusion-Personalization](https://github.com/PRIV-Creation/Awesome-Diffusion-Personalization)
- [Schuture/Benchmarking-Awesome-Diffusion-Models](https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models)
- [shogi880/awesome-controllable-stable-diffusion](https://github.com/shogi880/awesome-controllable-stable-diffusion)
- [Efficient Diffusion Models for Vision: A Survey](http://arxiv.org/abs/2210.09292)
- [Tracking Papers on Diffusion Models](https://vsehwag.github.io/blog/2023/2/all_papers_on_diffusion.html)

# License

This list is under the [Creative Commons licenses](https://creativecommons.org/) License.

