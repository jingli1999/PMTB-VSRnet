# PMTB-VSRnet

Official PyTorch implementation of **PMTB-VSRnet**, a physically guided video super-resolution (VSR) framework for enhancing Antarctic passive microwave brightness temperature (BT) sequences.

---

## Highlights

- Physically guided video super-resolution framework tailored for Antarctic AMSR2 brightness temperature sequences  
- Radiometrically coherent multi-temporal modeling via NCC-guided local alignment and bidirectional propagation  
- Residual-based reconstruction with asymmetric high-frequency constraint to ensure physical plausibility  
- Designed for single-channel 16-bit passive microwave brightness temperature data

---

## 1. Introduction

Passive microwave observations (e.g., AMSR2) provide all-weather monitoring capability over Antarctica and are essential for surface melt detection. However, their coarse spatial resolution (12.5–25 km) and antenna-induced blurring limit the representation of fine-scale radiometric transitions.

PMTB-VSRnet addresses this limitation by integrating radiometric coherence modeling and physics-driven regularization into a video super-resolution framework specifically designed for Antarctic brightness temperature sequences.

---

## 2. Method Overview

Given a low-resolution (LR) brightness temperature sequence {Y_t}, PMTB-VSRnet reconstructs the high-resolution (HR) sequence {X̂_t} under a fixed ×4 super-resolution setting.

The framework consists of:

### 2.1 Local Temporal Modeling

- Short window [t−1, t, t+1]
- Multi-scale PCD alignment
- Low-frequency NCC-guided adaptive fusion

### 2.2 Global Bidirectional Propagation

- Forward and backward recurrent branches
- Long-range temporal redundancy modeling

### 2.3 Residual-Based Reconstruction

- Bicubic-upsampled baseline
- Learned high-frequency residual
- Radiometric anchoring

### 2.4 Physically Guided Joint Loss

- Charbonnier reconstruction loss
- Asymmetric high-frequency constraint (Gaussian low-pass based)

The network is specifically adapted for 16-bit single-channel brightness temperature data.

---

## 3. Repository Structure

```
PMTB-VSRnet/
│
├── basicsr/
│   ├── train.py
│   ├── test.py
│   ├── archs/
│   ├── data/
│   ├── models/
│
├── options/
│
├── scripts/
│
├── results/   ← (optional, small demo only)
│
└── README.md
```

---

## 4. Environment Setup

Recommended:

- Python ≥ 3.8
- PyTorch ≥ 1.8
- CUDA-enabled GPU

```bash
conda create -n pmtb_vsrnet python=3.10
conda activate pmtb_vsrnet

pip install torch torchvision torchaudio
pip install numpy opencv-python pyyaml tqdm
```

---

## 5. Dataset Preparation

The experiments are based on publicly available AMSR2 Level-3 brightness temperature products provided by the Japan Aerospace Exploration Agency (JAXA).

AMSR2 data can be accessed and downloaded from:

- JAXA G-Portal:  
  https://gportal.jaxa.jp/

- Alternatively via GCOM-W1 AMSR2 product page:  
  https://gportal.jaxa.jp/gpr/?lang=en

In this study, Level-3 daily brightness temperature products (July 2012 – December 2024) at:

- 18.7 GHz (H/V polarization)
- 36.5 GHz (H/V polarization)

were used for Antarctic regions.

### Preprocessing

The following preprocessing steps were applied prior to model training:

1. Antarctic spatial subset extraction from global grids  
2. Conversion to 16-bit single-channel image format (PNG/TIF)  
3. Generation of low-resolution (LR) inputs using 4× bicubic downsampling  
4. Patch-based training with 256×256 crops (stride 128)

The processed HR–LR paired sequences are organized according to the directory structure described in Section 3.

Researchers can reproduce the dataset by following the above procedure using publicly available AMSR2 Level-3 products.

---


## 6. Publicly Available Super-Resolution Results

To facilitate reproducibility and comparison, the super-resolved outputs corresponding to the four channels reported in the paper are publicly available.

The released result folders include:

- PMTB_VSR_AMSR2_test_18.7H
- PMTB_VSR_AMSR2_test_18.7V
- PMTB_VSR_AMSR2_test_36.5H
- PMTB_VSR_AMSR2_test_36.5V

These datasets contain the ×4 super-resolved brightness temperature sequences used for quantitative evaluation in the paper.

Each result file is stored in 16-bit PNG format.

---

## 7. Implementation Details

- Optimizer: Adam (β1 = 0.9, β2 = 0.99)
- Initial learning rate: 2×10⁻⁴
- Cosine annealing schedule with warm-up
- Total iterations: 400,000
- Exponential moving average (EMA) decay: 0.999
- Training hardware: NVIDIA A800 GPUs

---

## 8. Training

Single GPU:

```bash
python basicsr/train.py -opt options/train/PMTB_VSRnet/train_PMTB_VSR_AMSR2-36.5H.yml
```

Multi-GPU:

```bash
./scripts/dist_train.sh 4 options/train/PMTB_VSRnet/train_PMTB_VSR_AMSR2-36.5H.yml
```

---

## 9. Testing

```bash
python basicsr/test.py -opt options/test/test_PMTB_VSR_AMSR2-36.5H.yml
```

---

## 10. Code Availability

The source code for PMTB-VSRnet is publicly available at:

https://github.com/jingli1999/PMTB-VSRnet

For reproducibility inquiries, please contact:  
2411781@tongji.edu.cn

---

## 11. Acknowledgements

This project is built upon:

- BasicSR
- BasicVSR
- EDVR

We sincerely thank the original authors for open-sourcing their frameworks.

---

## 12. License

This project is released under the same license as BasicSR.  
See the LICENSE file for details.