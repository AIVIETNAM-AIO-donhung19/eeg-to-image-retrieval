# EEG-to-Image Retrieval

*Exploring cross-modal EEG-to-image retrieval using contrastive learning and pretrained visual representations.*

## 1. Overview

This project investigates cross-modal image retrieval from non-invasive EEG signals using a pretrained CLIP embedding space.

Given an EEG recording collected while a subject views an image, the goal is to predict a 512-dimensional embedding from the EEG signal and retrieve the corresponding image from a candidate gallery based on cosine similarity.


The project uses a contrastive learning framework in which an EEG encoder is trained to align predicted EEG embeddings with pre-computed CLIP image embeddings through a symmetric InfoNCE loss.

The experiments focus on comparing different EEG encoder architectures and investigating the effects of repetition averaging during training and evaluation.

**Key highlights:**
- Cross-modal alignment between EEG signals and pretrained CLIP image embeddings.
- Comparison of MLP and 1D-CNN EEG encoders.
- Investigation of averaged (AVG) and non-averaged (Non-AVG) EEG representations.
- Evaluation on held-out visual concepts using a 200-image retrieval gallery.
- Analysis of retrieval performance and training dynamics under different experimental settings.

This project establishes an initial experimental baseline for EEG-to-image retrieval and provides a foundation for further investigation into EEG representations, model architectures, and contrastive learning strategies.


## 2. Dataset

### 2.1. Dataset Description

Experiments were conducted on a single subject (sub-03) from the THINGS-EEG2 dataset.

| Property | Description |
|---|---|
| Subject | sub-03 |
| EEG channels | 63 |
| Time samples | 100 |
| Training concepts | 1,323 |
| Validation concepts | 331 |
| Test concepts | 200 |
| Images per training/validation concept | 10 |
| Images per test concept | 1 |
| CLIP embedding dimension | 512 |

### 2.2 Data Splits

| Split | EEG Shape | Concepts | Images |
|---|---|---:|---:|
| Train | (13,230, 4, 63, 100) | 1,323 | 13,230 |
| Validation | (3,310, 4, 63, 100) | 331 | 3,310 |
| Test | (200, 80, 63, 100) | 200 | 200 |

Each training and validation image has four EEG repetitions, while each test image has 80 repetitions.

Training, validation, and test sets contain disjoint visual concepts, allowing the experiments to investigate retrieval performance on held-out concepts.

## 3. Methodology

### 3.1 Overall Pipeline

The proposed pipeline consists of five main stages:

1. **EEG preprocessing:** Process EEG signals into averaged or individual-repetition representations.
2. **CLIP embedding extraction:** Use a pretrained CLIP model to obtain 512-dimensional image embeddings.
3. **EEG encoding:** Map EEG signals into the same 512-dimensional embedding space using an MLP or Conv1D encoder.
4. **Contrastive alignment:** Train the EEG encoder using symmetric InfoNCE loss.
5. **Image retrieval:** Rank candidate images based on cosine similarity between predicted EEG embeddings and CLIP image embeddings.

### 3.2 EEG Preprocessing

Two EEG representations are investigated:

- **AVG:** Average four EEG repetitions corresponding to the same image during training.
- **Non-AVG:** Preserve individual EEG repetitions as separate training samples.

The same distinction is applied during evaluation:

- **Averaged test EEG:** Average 80 repetitions for each test image.
- **Single-trial test EEG:** Evaluate individual repetitions independently.

Per-channel Z-score normalization is applied using statistics computed from the corresponding training representation.

### 3.3 CLIP Image Embeddings

Pre-computed image embeddings are extracted using a pretrained CLIP model.

Each image is represented by a 512-dimensional vector, which serves as the target representation for EEG-to-image alignment.

The image embeddings are fixed during EEG encoder training.

### 3.4 Model Architectures

Two EEG encoder architectures are compared:

| Model | Parameters | Description |
|---|---:|---|
| MLP | 6,979,072 | Flattened EEG input followed by fully connected layers |
| Conv1D | 5,322,240 | Temporal convolutional feature extraction followed by fully connected layers |

Both models produce 512-dimensional EEG embeddings.

### 3.5 Contrastive Learning

A symmetric InfoNCE objective is used to align EEG embeddings with their corresponding CLIP image embeddings.

For each training batch, the model learns to maximize the similarity between matched EEG-image pairs while minimizing the similarity between mismatched pairs.

The main training configuration includes:

| Parameter | Value |
|---|---|
| Loss function | Symmetric InfoNCE |
| Batch size | 32 |
| Temperature | 0.07 |
| Optimizer | AdamW |
| Embedding dimension | 512 |
| Similarity metric | Cosine similarity |

## 4. Experimental Design

Four EEG encoders are trained using different combinations of architecture and training representation.

| Model | Architecture | Training representation |
|---|---|---|
| MLP_Avg | MLP | 4-repetition average |
| Conv1D_Avg | Conv1D | 4-repetition average |
| MLP_NonAvg | MLP | Individual repetitions |
| Conv1D_NonAvg | Conv1D | Individual repetitions |

Each trained model is evaluated on both averaged and single-trial test EEG, resulting in eight evaluation settings.

| Training | Architecture | Averaged Test | Single-trial Test |
|---|---|---|---|
| AVG | MLP | E1 | X-c |
| AVG | Conv1D | E2 | X-d |
| Non-AVG | MLP | X-a | E3 |
| Non-AVG | Conv1D | X-b | E4 |

This cross-condition evaluation allows the investigation of how training and evaluation representations influence retrieval performance.

## 5. Evaluation Metrics

Retrieval performance is evaluated using Recall@K.

For each EEG query, the predicted embedding is compared with the CLIP embeddings of 200 candidate test images using cosine similarity. The candidate images are then ranked according to their similarity scores.

Recall@K measures the proportion of queries for which the correct image appears among the top K retrieved images.

The chance levels for a gallery of 200 images are:

| Metric | Chance Level |
|---|---:|
| Recall@1 | 0.005 |
| Recall@5 | 0.025 |
| Recall@10 | 0.050 |

## 6. Experimental Results

### 6.1 Main Retrieval Results

The following table summarizes Recall@K across the four trained models under both test representations.

**Averaged test EEG (80-repetition average)**

| Model | Training | Recall@1 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| MLP | AVG | 0.150 | 0.380 | 0.545 |
| Conv1D | AVG | 0.150 | 0.410 | 0.565 |
| MLP | Non-AVG | 0.070 | 0.255 | 0.420 |
| Conv1D | Non-AVG | 0.055 | 0.215 | 0.400 |
| **Chance** | | **0.005** | **0.025** | **0.050** |

**Single-trial test EEG**

| Model | Training | Recall@1 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| MLP | AVG | 0.0284 | 0.1120 | 0.1909 |
| Conv1D | AVG | 0.0306 | 0.1247 | 0.2052 |
| MLP | Non-AVG | 0.0315 | 0.1165 | 0.2001 |
| Conv1D | Non-AVG | 0.0384 | 0.1563 | 0.2593 |
| **Chance** | | **0.005** | **0.025** | **0.050** |

### 6.2 Training and Validation Dynamics

Training and validation curves are analyzed to investigate convergence and generalization behavior.

The experiments show different learning dynamics across the four configurations:

- **AVG MLP:** Rapid overfitting, with validation Recall@5 peaking at epoch 3.
- **AVG Conv1D:** Validation performance peaks at epoch 7, followed by a decline.
- **Non-AVG MLP:** Moderate overfitting, with validation performance plateauing around 0.09 Recall@5.
- **Non-AVG Conv1D:** Validation Recall@5 continues improving through the final training epoch (100/100), suggesting that the training budget may be insufficient for convergence.

### 6.3 Key Findings

**1. Above-chance EEG-to-image retrieval**

All four models achieve retrieval performance above chance on both averaged and single-trial test EEG.

On averaged test EEG, Recall@5 ranges from 0.215 to 0.410, compared with a chance level of 0.025.

On single-trial test EEG, Recall@5 ranges from 0.112 to 0.156, also exceeding chance.

These results demonstrate that the learned EEG representations contain information that can be aligned with CLIP image embeddings for retrieval in this experimental setting.

**2. Consistency between training and evaluation representations**

Models trained on averaged EEG achieve better retrieval performance on averaged test EEG, whereas models trained on non-averaged EEG perform better on single-trial test EEG.

This suggests that matching the EEG representation between training and evaluation is important. The current results do not establish that averaging universally improves retrieval performance.

**3. Comparison between MLP and Conv1D**

Across all eight experimental settings, the Conv1D models consistently achieved better test retrieval performance than the MLP models. Notably, the fully connected layers of the Conv1D architecture were matched to those of the MLP, making the feature extraction stages the primary architectural difference. This result suggests that convolutional layers provide an effective way to extract and represent EEG features for the retrieval task.

## 7. Technical Report

For a detailed description of the experimental methodology, training configurations, learning curves, results, and discussion, see the [Technical Report](reports/technical_report.md).
