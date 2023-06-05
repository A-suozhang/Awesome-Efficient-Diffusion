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
    - b. Control Generation
    - c. Multi-Media Generation
- Deployment:
    - Ⅰ. GPU
    - Ⅱ. Mobile
    - Ⅲ. Miscellaneous Devices

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


### 0.3. Solvers

- [[ICLR21 - **DDIM**](https://arxiv.org/abs/2010.02502)]: "Denoising Diffusion Implicit Models";
    - determinstic sampling, 
    - reduce time-steps

- [[NeurIPS22 - **DPMSolver**](https://arxiv.org/abs/2206.00927)]: "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps";
    - follow-up work: [[Arxiv2211 - **DPMSolver++**](https://arxiv.org/abs/2211.01095)]: "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models";
    - multi-order ODE, faster convergence


### 0.4. Evaluation Metric

- [FID](https://huggingface.co/docs/diffusers/v0.16.0/en/conceptual/evaluation#quantitative-evaluation)

- [Clip-Score](https://huggingface.co/docs/diffusers/v0.16.0/en/conceptual/evaluation#quantitative-evaluation)

- [Other Metrics (Refering from Schuture/Benchmarking-Awesome-Diffusion-Models)](https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models)
    - **TODO**

### 0.5. Datasets & Settings

- **TODO**


## 1. Arch-level compression

> reduce the diffusion model cost (the u-net) with  *pruning* / *neural architecture search (nas)* techniques

# 🖨 Applications

## a. Personalization


## b. Controllable Generation


## c. Multi-modal Generation

# 🔋 Deployments

## Ⅰ. GPU

## Ⅱ. Mobile

## Ⅲ. Miscellaneous Devices

# Related

- [heejkoo/Awesome-Diffusion-Models](https://github.com/heejkoo/Awesome-Diffusion-Models)
- [awesome-stable-diffusion/awesome-stable-diffusion](https://github.com/awesome-stable-diffusion/awesome-stable-diffusion)
- [hua1995116/awesome-ai-painting](https://github.com/hua1995116/awesome-ai-painting)
- [PRIV-Creation/Awesome-Diffusion-Personalization](https://github.com/PRIV-Creation/Awesome-Diffusion-Personalization)
- [Schuture/Benchmarking-Awesome-Diffusion-Models](https://github.com/Schuture/Benchmarking-Awesome-Diffusion-Models)
- [shogi880/awesome-controllable-stable-diffusion](https://github.com/shogi880/awesome-controllable-stable-diffusion)

# License

This list is under the [Creative Commons licenses](https://creativecommons.org/) License.

