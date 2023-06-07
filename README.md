# Awesome Efficient Diffusion

[![Awesome](https://awesome.re/badge-flat.svg)](https://awesome.re)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com)

> A curated list of methods that focus on improving the efficiency of diffusion models

### Index

- Algorithms:
    - 0. Basics
    - 1. Arch-level Compression
    - 2. Time-step-level Compression
    - 3. Data-precision-level Compression
    - 4. Input-level Compression
    - 5. Efficient Tuning
- Applications:
    - a. Personalization
    - b. Controllable Generation
    - c. Multi-Media Generation
- Deployment:
    - Ⅰ. GPU
    - Ⅱ. Mobile
    - Ⅲ. Miscellaneous Devices

### Design Principles

- **Simple**: Summarize structural points as paper description, omit the details (dont get lost in low information texts)
- **Quantitative**: give the relative speedup for certain method (how far have we come?)

# 🔬 Algotithms

## 0. Basics

> Some basic diffusion model papers, specifying the preliminaries. Noted that the main focus of this awesome list is the efficient method part. Therefore it only contains minimum essential preliminary studies you need to know before acclerating diffusion. 

> The [[🤗 Huggingface Diffuser Doc](https://huggingface.co/docs/diffusers/index)] is also a good way of getting started with diffusion

### 0.1. Methods (Variants/Techniques)

> some famous diffusion models (e.g., DDPM, Stable-Diffusion), and key techniques

- [[ICML15 - **DPM**](https://arxiv.org/abs/1503.03585)] : "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"; 
    - Pre-DDPM, early diffusion model

- [[NeurIPS20 - **DDPM**](https://arxiv.org/abs/2006.11239)]: "Denoising Diffusion Probabilistic Models"
    - Discrete time diffusion model

- [[ICLR21 - **SDE**](https://arxiv.org/abs/2011.13456)]: "Score-Based Generative Modeling through Stochastic Differential Equations";
    - Continuous time Neural SDE formulation of diffusion

- [[Arxiv2105 -  **Classifier-Guidance**](https://arxiv.org/abs/2105.05233)]: "Diffusion Models Beat GANs on Image Synthesis"; 
    - Conditional generation with classifier guidance

- [[CVPR22 -  **Stable-Diffusion**](2112.10752)]: "High-Resolution Image Synthesis with Latent Diffusion Models";
    - Latent space diffusion (with VAE)
    - Latent class guidance (with CLIP embedding fed into cross_attn)
    - [[📎 Code:CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)]

- [[TechReport-2204 - **DeepFlyoid-IF**](https://github.com/deep-floyd/IF)] ;
    - following [[NeurIPS22 - **Imagen**](https://arxiv.org/abs/2205.11487)]: “Photorealistic Text-to-Image Diffusion Models
with Deep Language Understanding”;
    - Larger Language Model (T5 over CLIP)
    - Pixel-space Diffusion
    - Diffusion for SR
    - [[📎 Code: sdeep-floyd/IF](https://github.com/deep-floyd/IF)]

### 0.2. Architecture Components

- *"Vision-Language Model"*
    - [CLIP](https://arxiv.org/abs/2103.00020), [T5](https://arxiv.org/abs/1910.10683)
    - *Containing Operations:*
        - Self-Attention (Cross-Attention)
        - FFN (FC)
        - LayerNorm (GroupNorm)
- *Diffusion Model* (also sometimes used for SuperReoslution)
    - [U-Net](https://arxiv.org/abs/1505.04597)
    - *Containing Operations:*
        - Conv
        - DeConv (ConvTransposed, Interpolation)
        - Low-range Shortcut Connection
- *Encoder-Decoder* (for latent-space diffusion)
    - [VAE](https://arxiv.org/abs/1606.05908) (in latent diffusion)
    - *Containing Operations:*
        - Conv
        - DeConv (ConvTransposed, Interpolation)

### 0.3. Solver (Sampler)

- [[ICLR21 - **DDIM**](https://arxiv.org/abs/2010.02502)]: "Denoising Diffusion Implicit Models";
    - determinstic sampling, 
    - reduce time-steps

- [[NeurIPS22 - **DPMSolver**](https://arxiv.org/abs/2206.00927)]: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps";
    - follow-up work: [[Arxiv2211 - **DPMSolver++**](https://arxiv.org/abs/2211.01095)]: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models";
    - multi-order ODE, faster convergence

 - ![](https://img.shields.io/static/v1?message=todo&color=red)

### 0.4. Evaluation Metric

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

 - ![](https://img.shields.io/static/v1?message=todo&color=red)

### 0.5. Datasets & Settings

#### 0.5.1 Unconditional Generation

#### 0.5.2 Text-to-Image Generation

- **CIFAR-10**: 

- **CelebA**:
#### 0.5.3 Image/Depth-to-Image Generation

- ![](https://img.shields.io/static/v1?message=todo&color=red)



## 1. Arch-level compression

> reduce the diffusion model cost (the repeatedly inference u-net) with  *pruning* / *neural architecture search (nas)* techniques

- [[Arxiv2305](https://arxiv.org/abs/2305.10924)] "Structural Pruning for Diffusion Models";
    - [Code](https://github.com/VainF/Diff-Pruning)

## 2. Timestep-level Compression

> reduce the timestep (the number of u-net inference)

### 2.1 Improved Sampler

> Improved sampler, faster convergence, less timesteps

- [[ICLR21 - **DDIM**](https://arxiv.org/abs/2010.02502)]: "Denoising Diffusion Implicit Models";
    - **📊 典型结果**：50~100 Steps -> 10~20 Steps with moderate perf. loss
- [[NeurIPS22 - **DPMSolver**](https://arxiv.org/abs/2206.00927)]: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps";
    - **📊 典型结果**：NFE(num of unet forward)=10 achieves similar performance with DDIM NFE=100
### 2.2 Improved Training 

> Distillation/New Scheme


- [[Arxiv2305 - **CatchUpDistillation**](https://arxiv.org/abs/2305.10769)]: "Catch-Up Distillation: You Only Need to Train Once for Accelerating Sampling";

- [[ICML23 - **ReDi**](https://arxiv.org/abs/2302.02285)]: "ReDi: Efficient Learning-Free Diffusion Inference via Trajectory Retrieval";
    - Skip intemediate steps:
    - Retrieval: find similar partially generated scheduling in early stage

- [[Arxiv2303 - **Consistency Model**](https://arxiv.org/pdf/2303.01469.pdf)]: "Consistency Models";
    - New objective: consistency based



## 3. Data-precision-level Compression

> quantization & low-bit inference/training

- [[Arxiv2305 - **PTQD**](https://arxiv.org/abs/2305.10657)]: "PTQD: Accurate Post-Training Quantization for Diffusion Models";

- [[Arxiv2304 - **BiDiffusion**](https://arxiv.org/abs/2304.04820)] "Binary Latent Diffusion"; 





## 4. Input-level Compression

### 4.1 Adaptive Inference

> save computation for different sample condition (noise/prompt/task)

- [[Arxiv2304 - **ToMe**](https://arxiv.org/abs/2303.17604)]: "Token Merging for Fast Stable Diffusion";
### 4.2 Patched Inference

> reduce the processing resolution

- [[Arxiv2304 - **PatchDiffusion**](https://arxiv.org/abs/2304.12526)]: "Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models"

- [[CVPR23W - **MemEffPatchGen**](https://arxiv.org/abs/2304.07087)]: "Memory Efficient Diffusion Probabilistic Models via Patch-based Generation";

## 5. Efficient Tuning

- [[Arxiv2304 - **DiffFit**](https://arxiv.org/abs/2304.06648)]: "DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning"

- [[Arxiv2303 - **ParamEffTuningSummary**](https://arxiv.org/abs/2303.18181)]: "A Closer Look at Parameter-Efficient Tuning in Diffusion Models";
### 5.1. Low-Rank 

> The LORA family




# 🖨 Applications

## a. Personalization


## b. Controllable Generation


## c. Multi-modal Generation

# 🔋 Deployments

## Ⅰ. GPU

## Ⅱ. Mobile

- [[Arxiv2306 - **SnapFusion**](http://arxiv.org/abs/2306.00980)]: "SnapFusion: Text-to-Image Diffusion Model on Mobile Devices within Two Seconds";
    - Platform: Iphone 14 Pro, 1.84s
    - Model Evolution: 3.8x less param compared with SD-V1.5
    - Step Distilaltion into 8 steps

## Ⅲ. Miscellaneous Devices

# Related

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

