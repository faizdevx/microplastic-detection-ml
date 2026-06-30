# Microplastic Detection and Quantification using Machine Learning and Computer Vision

A low-cost, portable, AI-assisted system for quantification and polymer classification of microplastics in water using multi-angle light scattering, fluorescence imaging, and deep learning.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![YOLOv11](https://img.shields.io/badge/YOLOv11-InstanceSegmentation-green)
![License](https://img.shields.io/badge/License-MIT-orange)

---

# Overview

Microplastic pollution has become a major environmental challenge worldwide. Traditional methods such as FTIR, Raman spectroscopy, and Py-GC/MS provide excellent accuracy but require expensive laboratory infrastructure and trained personnel.

This project proposes a low-cost, field-deployable microplastic detection platform that combines:

- Multi-angle light scattering (MALS)
- Fluorescence imaging using Nile Red staining
- Computer vision-based polymer identification
- Machine learning-based concentration estimation
- Real-time edge deployment capability

The system aims to detect and quantify microplastic particles ranging from **10–100 μm** while providing polymer classification using deep learning.

---

# Repository

### GitHub Repository

https://github.com/faizdevx/microplastic-detection-ml

---

# Dataset and Research Archive

The complete dataset, supplementary materials, training configurations, and experimental records are archived on Zenodo.

### Zenodo Archive

https://zenodo.org/records/21073653

### DOI

```
10.5281/zenodo.21073653
```

---

# Research Paper

### Title

**Machine Learning based algorithm for quantification and detection of different types of Microplastic in Water Sample**

### Authors

- Gaurav Joshi
- Shashwat Singh
- Shiv Kumar Gupta
- Faisal Arif Ansari
- Aviral Madhvan
- Dhirendra Kumar Sharma
- Sweta Shukla

---

# System Architecture

The proposed system consists of two major subsystems:

## 1. Quantification Module

Uses:

- 650 nm laser source
- Multi-angle scattering
- BPW34 photodiodes
- Mie scattering analysis
- Machine learning regression

```
Water Sample
      ↓
Laser Scattering
      ↓
Photodiode Measurements
      ↓
Feature Extraction
      ↓
ML Regression
      ↓
Microplastic Concentration
```

---

## 2. Polymer Identification Module

Uses:

- Nile Red fluorescence staining
- 365 nm UV excitation
- Optical filtering
- Raspberry Pi Global Shutter Camera
- YOLOv11 instance segmentation

```
Microplastic Sample
        ↓
Nile Red Staining
        ↓
UV Excitation
        ↓
Fluorescence Imaging
        ↓
YOLOv11 Segmentation
        ↓
Polymer Classification
```

---

# Hardware Components

| Component | Specification |
|-----------|--------------|
| Laser | 650 nm |
| UV LED | 365 nm |
| Camera | Raspberry Pi Global Shutter Camera |
| Sensor | BPW34 Photodiodes |
| Lens | 10× Objective |
| Controller | Raspberry Pi 5 |
| Optical Filter | 520 nm Bandpass Filter |
| Detection Range | 10–100 μm |

---

# Machine Learning Models

## Concentration Estimation

Models evaluated:

- Linear Regression
- Lasso Regression
- Random Forest Regression
- Support Vector Regression (RBF)

Features:

```
S1
S2
S3
S4
S_mean
S_std
Forward/Backward Ratio
Scattering Ratios
```

---

## Polymer Classification

Model:

```
YOLOv11 Instance Segmentation
```

Classes:

- PVC
- Nylon

Training configuration:

| Parameter | Value |
|---|---|
| Input Size | 300×300 |
| Epochs | 100 |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Dataset Split | 70/20/10 |
| Annotation | Polygon Segmentation |

---

# Dataset Statistics

| Polymer | Unique Images |
|----------|--------------|
| Nylon | 1085 |
| PVC | 1040 |
| Total | 2125 |

Augmentation techniques:

- Rotation
- Scaling
- Brightness adjustment
- Gaussian noise
- Horizontal flipping
- Vertical flipping

---

# Computer Vision Performance

| Polymer | mAP@50 (Box) | mAP@50 (Mask) |
|---|---|---|
| PVC | 0.92 | 0.90 |
| Nylon | 0.94 | 0.92 |

Additional metrics:

- Precision
- Recall
- F1-score
- Confusion Matrix
- Failure Case Analysis

---

# Optical Physics

The concentration estimation module is based on:

- Beer-Lambert Law
- Mie Scattering Theory
- Multi-Angle Light Scattering
- Forward Scattering Asymmetry
- Multiple Scattering Correction

For particles ranging from:

```
10–100 μm
```

and laser wavelength:

```
650 nm
```

the dominant scattering mechanism becomes:

```
Mie Scattering
```

which exhibits strong forward scattering characteristics.

---

# Project Structure

```bash
microplastic-detection-ml/
│
├── datasets/
├── notebooks/
├── models/
├── training/
├── results/
├── figures/
├── supplementary/
├── paper/
└── README.md
```

---

# Installation

```bash
git clone https://github.com/faizdevx/microplastic-detection-ml.git

cd microplastic-detection-ml

pip install -r requirements.txt
```

---

# Citation

If you use this repository or dataset, please cite:

```bibtex
@dataset{ansari2026microplastic,
  author = {Ansari, Faisal Arif and others},
  title = {Machine Learning Based Microplastic Detection Dataset},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.21073653},
  url = {https://zenodo.org/records/21073653}
}
```

---

# Links

### GitHub Repository

https://github.com/faizdevx/microplastic-detection-ml

### Zenodo Dataset

https://://zenodo.org/records/21073653

---

# Future Work

- Environmental weathered microplastic datasets
- Explainable AI (XAI)
- Spectral imaging
- Edge AI deployment
- Multi-polymer classification
- Real-time river monitoring
- Autonomous sensing probes

---

# License

Apache License
