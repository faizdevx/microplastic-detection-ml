# microplastic-detection-ml


# Response to Reviewer Comments

**Manuscript Title:**  
**Machine Learning based algorithm for quantification and detection of different types of Microplastic in Water Sample**

**Authors:**  
Gaurav Joshi, Shashwat Singh, Shiv Kumar Gupta, Faisal Arif Ansari, Aviral Madhvan, Dhirendra Kumar Sharma, Sweta Shukla

---

## General Response

We sincerely thank the reviewers for their exceptionally detailed, constructive, and technically rigorous comments. Their observations have significantly strengthened the scientific quality, experimental validation, and machine learning methodology presented in our work. In response, we have revised the manuscript extensively by:

- Expanding the machine learning validation framework.
- Incorporating strict data leakage prevention procedures.
- Providing detailed dataset composition and annotation methodology.
- Implementing and benchmarking non-linear regression models.
- Expanding discussions on explainable and interpretable machine learning.
- Improving the robustness analysis of the computer vision pipeline against environmental microplastic variability.
- Updating the discussion of environmental weathering effects and model generalization.

A point-by-point response to all machine learning related comments is provided below.

---

# Response to Reviewer 1

---

## Comment 4

> The machine-learning analysis needs stronger validation. The authors should report dataset size, image annotation procedure, train/validation/test split, augmentation strategy, hyperparameters, confusion matrix, precision, recall, F1-score, mAP, and representative failure cases. Potential data leakage between training and testing datasets should also be excluded.

### Response

We fully agree with the reviewer that rigorous validation is essential for establishing the reliability and reproducibility of machine learning models.

To address this concern, the machine learning methodology section has been substantially expanded.

### 1. Prevention of Data Leakage

To completely eliminate data leakage, the dataset splitting procedure was performed **before any augmentation step**. The dataset was partitioned at the level of the original microscopy frames using a strict **7:2:1 train/validation/test split**. Consequently, augmented variants derived from a particular particle image were constrained to remain exclusively within the same dataset partition.

This protocol guarantees that no synthetically generated variation of a particle appears simultaneously in both training and testing datasets, thereby preventing overly optimistic performance estimates.

---

### 2. Dataset Composition

The foundational fluorescence microscopy dataset consisted of independently captured microplastic particles acquired using the Raspberry Pi Global Shutter Camera equipped with a 10× objective lens.

The curated dataset contains:

| Polymer | Unique Images |
|----------|--------------|
| Nylon | 1,085 |
| PVC | 1,040 |
| Total | 2,125 |

These images represent independent particle captures prior to any augmentation procedures.

The dataset includes variability in:

- particle geometry,
- fragmentation patterns,
- fluorescence intensity,
- surface roughness,
- illumination variation,
- imaging noise,
- staining variability.

This diversity was specifically incorporated to improve model robustness and reduce overfitting.

---

### 3. Annotation Strategy

All particle annotations were performed manually using **Roboflow polygon segmentation tools**.

Polygon masks were intentionally selected instead of rectangular bounding boxes because:

- microplastic particles possess highly irregular morphologies,
- fluorescence signatures are distributed internally,
- segmentation masks better preserve particle texture information,
- polygon annotations minimize inclusion of irrelevant background pixels.

This approach allowed the model to learn not only particle morphology but also the internal fluorescence distribution patterns critical for polymer discrimination.

---

### 4. Data Augmentation Strategy

To improve model generalization capability, geometric and photometric augmentation strategies were employed. The transformation space included:

- Rotation
- Scaling
- Brightness adjustment
- Gaussian noise injection
- Horizontal/vertical flipping
- Controlled cropping

Mathematically:

$$
x' \sim T(x)
$$

where:

- $x$ represents the original image,
- $x'$ represents the augmented image,
- $T$ denotes the transformation space.

The augmentation pipeline acted as an implicit regularizer and significantly reduced overfitting.

---

### 5. Training Configuration

The YOLOv11 instance segmentation model was trained using:

| Parameter | Value |
|---|---|
| Input size | 300 × 300 |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Epochs | 100 |
| Batch size | 16 |
| Weight initialization | Random |
| Early stopping | Validation mAP plateau |

Training was performed entirely on Google Colab.

The model was trained from scratch because fluorescence microscopy images differ substantially from natural image datasets typically used for transfer learning.

---

### 6. Performance Metrics

The revised manuscript now includes the following evaluation metrics:

| Polymer | mAP@50 (Box) | mAP@50 (Mask) | Precision | Recall | F1-score |
|---|---|---|---|---|---|
| PVC | 0.92 | 0.90 | 0.91 | 0.89 | 0.90 |
| Nylon | 0.94 | 0.92 | 0.93 | 0.91 | 0.92 |

Additional evaluation materials have been added, including:

- confusion matrices,
- precision-recall curves,
- F1-score curves,
- segmentation mask visualization,
- representative failure examples.

---

### 7. Failure Case Analysis

Representative failure cases have been included in the supplementary materials. Typical failures occur under:

- severe fluorescence bleaching,
- particle diameters below approximately 15 μm,
- low signal-to-noise conditions,
- highly irregular weathered fragments,
- environmental debris exhibiting fluorescence characteristics similar to stained microplastics.

These cases provide insight into the current limitations of the proposed methodology.

---

## Comment 8

> The reference should be improved. The authors may also consider discussing recent progress in interpretable machine learning for microplastic-related systems.

### Response

We appreciate the reviewer's valuable recommendation regarding interpretable machine learning.

The revised manuscript now includes a dedicated discussion on explainable machine learning approaches and their importance in environmental sensing applications. In particular, we discuss how our multi-angle scattering model remains physically interpretable because the predictor variables directly correspond to measurable optical quantities such as:

- forward scattering intensity,
- irradiance measurements,
- Mie scattering behavior,
- optical path interactions.

Unlike many black-box deep learning approaches, our regression framework allows explicit interpretation of feature importance and physical relationships.

The suggested article:

> *Elucidating microplastic adsorption mechanisms in biomass composite materials through interpretable machine learning*, Journal of Hazardous Materials, 2026, 501:140700

has been incorporated into the revised discussion.

---

# Response to Reviewer 2

---

## Comment 9 & 11

> Training exclusively on pristine commercial plastics introduces dataset bias.

### Response

We fully agree that environmental weathering can substantially alter microplastic morphology and optical behavior.

### 1. Dataset Clarification

The originally reported dataset size represented only a subset of the entire imaging dataset.

The complete dataset consists of:

- 1,085 unique Nylon particle images,
- approximately 1,040 unique PVC particle images,

resulting in over **2,100 unique particle images before augmentation**.

---

### 2. Environmental Generalization

To mitigate dataset bias introduced by pristine laboratory samples, we expanded our dataset using:

- UV-C induced photo-oxidative aging,
- controlled mechanical abrasion,
- surface weathering protocols,
- environmental water spiking experiments.

These procedures introduced:

- surface roughness,
- partial transparency,
- fluorescence heterogeneity,
- shape irregularity,
- biofouling-like characteristics.

Performance evaluation demonstrated strong robustness under these modified conditions.

---

## Comment 12 & 13

> The lasso regression model forces a linear relationship onto multi-angle scattering data, which is fundamentally non-linear.

### Response

We completely agree with the reviewer's assessment.

At elevated concentrations, optical interactions become dominated by:

- multiple scattering,
- particle shadowing,
- phase interference,
- detector saturation effects.

Consequently, linear assumptions become insufficient.

### Previous Approach

Our initial model employed:

$$
\bar{ADC}=\frac{S_1+S_2+S_3+S_4}{4}
$$

combined with Lasso regression due to its ability to handle multicollinearity and suppress sensor noise.

### Updated Non-Linear Models

To better capture the underlying optical physics, we extracted the complete multi-angle feature vector:

$$
X=[S_1,S_2,S_3,S_4]
$$

and implemented:

- Support Vector Regression (RBF kernel),
- Random Forest Regression,
- Grid Search hyperparameter optimization.

The RBF-SVR demonstrated excellent performance in modeling non-linear scattering transfer functions, while Random Forest models effectively captured complex interactions between sensor channels.

The revised manuscript now includes:

- model comparison tables,
- cross-validation results,
- hyperparameter optimization details,
- updated Figure 15,
- expanded Section 4.2.3 discussion.

---

## Comment 15

> The recent works on hybrid modelling and augmented life-cycle assessment should be discussed.

### Response

We thank the reviewer for these important recommendations.

The revised manuscript now discusses recent advances in:

- hybrid modeling architectures,
- augmented life-cycle assessment,
- environmental digital twins,
- machine-learning-assisted environmental monitoring.

These studies reinforce the motivation behind our work: combining low-cost sensing hardware with machine learning frameworks to develop scalable environmental monitoring systems suitable for field deployment.

---

# Final Statement

We sincerely thank both reviewers for their constructive and scientifically rigorous feedback. Their comments have substantially improved the quality, transparency, and technical depth of our manuscript. We believe the revised manuscript now provides a significantly stronger experimental foundation, a more rigorous machine learning methodology, and a clearer discussion of the limitations and future directions of low-cost microplastic detection systems.

---

**Submitted on behalf of all authors.**
