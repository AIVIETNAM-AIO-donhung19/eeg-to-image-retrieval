# Technical Report: EEG-to-Image Retrieval with Pretrained CLIP Embeddings

---

## 1. Abstract

We analyze a set of experiments on a single subject (sub-03, 63 EEG channels × 100 time samples) in which EEG is mapped into 512-d CLIP image-embedding space with a symmetric InfoNCE loss and evaluated by closed-set 200-way image retrieval. Four models were trained: an MLP and a 1D-CNN, each trained either on 4-repetition-averaged EEG ("AVG") or on single-trial EEG ("Non-AVG"). Each model was tested on two test sets: EEG averaged over 80 repetitions and single-trial EEG, giving 8 evaluations. **On the 80-rep-averaged test set, AVG-trained models reach Recall@5 of 0.380 (MLP) and 0.410 (CNN), against 0.255 and 0.215 for Non-AVG-trained models (chance 0.025).** **On single-trial test data, all four models fall to Recall@5 0.112–0.156, and the best model is the Non-AVG-trained CNN (0.156)**. The learning curves indicate **strong overfitting for the AVG models** after epoch 3 (MLP) and epoch 7 (CNN), **moderate overfitting for the Non-AVG MLP, and an unconverged Non-AVG CNN** whose best epoch is the last one (100/100).

---

## 2. Introduction and Research Questions

### 2.1 Problem and motivation

This project focuses on cross-modal retrieval. Given an EEG signal recorded while a subject views an image, the model predicts an embedding from the EEG signal. It then compares this embedding with the CLIP embeddings of candidate images using cosine similarity and ranks the images accordingly.

The motivation is to explore whether visual information can be retrieved from non-invasive EEG recordings, which typically contain noisy signals, by using a pretrained CLIP embedding space. Instead of classifying images into a fixed set of categories, this approach allows the model to match EEG signals with images based on their semantic representations.

### 2.2 Research questions

- **RQ1:** Do the four trained models retrieve the correct test image above chance?
  Evidence: E1–E4 absolute Recall@K versus chance K/200.
- **RQ2 (non-average / averaging).** Does averaging EEG repetitions (at training and/or test time) improve retrieval, and is training on non-averaged repetitions beneficial?
  Evidence: E1/E2 versus E3/E4, plus the four cross-condition evaluations that the notebooks also compute.
- **RQ3 (architecture).** Does a Conv1D encoder outperform an MLP encoder?
  Evidence: E1 vs E2 (averaged test), E3 vs E4 (single-repetition test).

**Cross-cutting assumption:** Generalisation to unseen visual concepts is implicit in the design (validation concepts are disjoint from training; test concepts are presumed disjoint).

### 2.3 Mapping of the four experiments

The task description defines E1–E4 as {MLP, Conv1D} × {averaged test, non-averaged test}. In the notebooks the four *models* are `MLP_Avg`, `Conv1D_avg`, `MLP_NonAvg`, `Conv1d_NonAvg` (named by **training** regime), and each is evaluated on **both** test representations (8 evaluations in total). This report uses:

| ID | Model | Training data | Test representation | 
|---|---|---|---|
| E1 | MLP | 4-repetition average | 80-repetition average | AVG nb, cell 65 |
| E2 | Conv1D | 4-repetition average | 80-repetition average | AVG nb, cell 65 |
| E3 | MLP | 4 individual repetitions | 80 individual repetitions | NON-AVG nb, cell 69 |
| E4 | Conv1D | 4 individual repetitions | 80 individual repetitions | NON-AVG nb, cell 69 |
|
| X-a | MLP | 4 individual repetitions | 80-repetition average | NON-AVG nb, cell 69 |
| X-b | Conv1D | 4 individual repetitions | 80-repetition average | NON-AVG nb, cell 69 |
| X-c | MLP | 4-repetition average | 80 individual repetitions | AVG nb, cell 65 |
| X-d | Conv1D | 4-repetition average | 80 individual repetitions | AVG nb, cell 65 |
---

## 3. Dataset Description


| Split | EEG tensor | Image embeddings | Concepts | Images/concept | EEG/image |
|---|---|---|---|---|---|
| Train | (13230, 4, 63, 100) | (13230, 512) | 1323 | 10 | 4 sessions |
| Validation | (3310, 4, 63, 100) | (3310, 512) | 331 | 10 | 4 sessions |
| Test | (200, 80, 63, 100) | (200, 512) | 200 | 1 | 80 sessions |

---

## 4. Methodology

### 4.1 EEG preprocessing (in-notebook operations)

2. **AVG pipeline:** mean over dimension 1: train `(16540,4,63,100) → (16540,63,100)` (mean of 4 repetitions); test `(200,80,63,100) → (200,63,100)` (mean of 80 repetitions).
3. **Non-AVG pipeline:** train `(13230,4,63,100) → reshape (52920,63,100)`; CLIP targets and concept IDs expanded with `repeat_interleave(4)`; validation kept as `(3310,4,63,100)` and evaluated per repetition; test reshaped to `(16000,63,100)` with targets repeated 80× .
4. **Normalization:** Using per-channel Z-Scoring. The AVG pipeline uses statistics of the 4-averaged training data; the Non-AVG pipeline uses statistics of the 52,920 non-averaged samples.

### 4.2 CLIP image embeddings

Image embeddings are pre-computed 512-d vectors loaded from `.npy` files. One embedding per training/validation image (16,540) and per test image (200). 

### 4.3 Architectures

**MLP** (`MLP`, 6,979,072 parameters ):

`(B,63,100) → Flatten (6300) → Linear(6300,1024) → LayerNorm → ReLU → F.dropout(p=0.4) → Linear(1024,512)`

**Conv1D** (`ImprovedConv1d_v2`, 5,322,240 parameters ), channels as Conv1D input channels, convolution over time:

| Stage | Operation | Output |
|---|---|---|
| Input | | (B, 63, 100) |
| Block 1 | Conv1d(63→128, k=5, pad=2) → BN → ReLU → AvgPool(2) | (B, 128, 50) |
| Block 2 | Conv1d(128→256, k=5, pad=2) → BN → ReLU → AvgPool(2) | (B, 256, 25) |
| Block 3 | Conv1d(256→512, k=3, pad=1) → BN → ReLU → AdaptiveAvgPool(8) | (B, 512, 8) |
| Head | Flatten (4096) → Linear(4096,1024) → LayerNorm → ReLU → Dropout(0.4) → Linear(1024,512) | (B, 512) |

**Initialisation.** He-normal initialisation (`init_weights_he`) is applied to the MLP and Conv1D models. 

### 4.4 Contrastive learning

For a batch of size *B* = 32, with EEG embeddings *z_i* and CLIP embeddings *v_i*, both L2-normalised, and fixed temperature *τ* = 0.07:

$$ s_{ij} = \frac{z_i^\top v_j}{\tau}, \qquad
\mathcal{L} = \tfrac{1}{2}\Big[\tfrac1B\sum_i -\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ij}}} \;+\; \tfrac1B\sum_i -\log\frac{e^{s_{ii}}}{\sum_j e^{s_{ji}}}\Big] $$

**Batch construction differs between pipelines [Fact]:**
- **AVG:** standard `DataLoader(shuffle=True, batch_size=32)` over 13,230 samples (413 steps/epoch). `UniqueConceptBatchSampler` is defined but not used. With 10 images per concept, a random batch contains on average ≈0.34 same-concept pairs (496 pairs × 9/13,229), which are treated as negatives.
- **Non-AVG:** `UniqueConceptBatchSampler`: each epoch draws **one random (image, repetition) sample per concept** (1,323 samples/epoch → 41 steps/epoch, `drop_last=True`), so every batch has unique concepts.

---

## 5. Experimental Setup

| Setting | AVG models | Non-AVG models |
|---|---|---|
| Training samples available | 13,230 | 52,920 |
| Samples actually seen per epoch | 13,230 | 1,323 (one per concept) |
| Steps per epoch | 413 | 41 |
| Epochs | 15 | 100 |
| Optimiser | AdamW | AdamW |
| LR | 1e-3 | 5e-3 |
| Weight decay | MLP: not set; Conv1D: 1e-4 | MLP: not set; Conv1D: 1e-4 |
| Batch size | 32 | 32 |
| Temperature | 0.07  | 0.07 |
| Checkpoint criterion | max validation Recall@5 | max validation Recall@5 (mean over 4 repetitions) |
| Selected epoch | MLP: 3/15; Conv1D: 7/15 | MLP: 95/100; Conv1D: 100/100 |



---

## 6. Evaluation Metrics

Given L2-normalised predicted embeddings *ẑ_i* and gallery embeddings *v_j*, similarity is cosine, `S_ij = ẑ_i · v_j`. Rank of the correct image for query *i*:

$$ \text{rank}_i = 1 + \#\{j \neq i : S_{ij} > S_{ii}\}, \qquad \text{Recall@}K = \frac1N \sum_i \mathbb{1}[\text{rank}_i \le K] $$

Chance level with *N* gallery images is *K/N*: **0.005 / 0.025 / 0.050** for K = 1/5/10 with *N* = 200.


---

## 7. Experimental Results

### 7.1 Training and validation dynamics

| Model | Selected epoch | Val Recall@5 at selection | Chance (N=331) | Final train loss | Final val loss | Final train Recall@5 | Observation |
|---|---|---|---|---|---|---|---|
| MLP, AVG | 3/15 | 0.1615 ± 0.016 | 0.015 | 0.272 | 3.222 | 0.985 | Rapid overfitting: val R@5 falls to 0.118 by epoch 15 |
| Conv1D, AVG | 7/15 | 0.2293 ± 0.020 | 0.015 | 0.484 | 3.089 | 0.884 | Peak at epoch 7, then declines to 0.174 |
| MLP, Non-AVG | 95/100 | 0.0941 (mean of 4 reps) | 0.015 | 2.256 | 3.106 | 0.203 | Val R@5 plateaus around 0.09; val loss minimum at epoch 80 (3.04) |
| Conv1D, Non-AVG | 100/100 | 0.1250 (mean of 4 reps) | 0.015 | 2.680 | 2.854 | 0.063 | Still improving at epoch 100 (not converged) |

The AVG models show signs of overfitting after only a few epochs: their training Recall@5 approaches 1.0, while their validation Recall@5 starts to decrease. In contrast, the Non-AVG models exhibit less overfitting. However, the validation Recall@5 of the Non-AVG Conv1D model was still increasing at the final epoch. Therefore, its selected checkpoint reflects the limited training budget rather than the model having fully converged.


### 7.2 Test results

**Averaged test EEG**

| Experiment | Model | Training | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| E1 | MLP | AVG | 0.150 | 0.380 | 0.545 |
| E2 | Conv1D | AVG | 0.150 | 0.410 | 0.565 |
| X-a | MLP | Non-AVG | 0.070 | 0.255 | 0.420 |
| X-b | Conv1D | Non-AVG | 0.055 | 0.215 | 0.400 |
| *Chance* | | | *0.005* | *0.025* | *0.050* |

**Single-repetition test EEG**

|Experiment| Model | Training | Recall@1 | Recall@5 | Recall@10 |
|---|---|---|---|---|---|
| X-c | MLP | AVG | 0.0284 | 0.1120 | 0.1909 |
| X-d | Conv1D | AVG | 0.0306 | 0.1247 | 0.2052 |
| E3| MLP | Non-AVG | 0.0315 | 0.1165 | 0.2001 |
| E4| Conv1D | Non-AVG | 0.0384 | 0.1563 | 0.2593 |
| *Chance* || | *0.005* | *0.025* | *0.050* |


- **Above chance:** All models achieved retrieval performance above the chance level on both averaged and single-repetition test EEG. For averaged test EEG, Recall@1 ranged from 0.055 to 0.150, compared with a chance level of 0.005. Recall@5 ranged from 0.215 to 0.410, substantially exceeding the chance level of 0.025. For single-repetition test EEG, Recall@1 ranged from 0.0284 to 0.0384, while Recall@5 ranged from 0.1120 to 0.1563, also remaining above chance.

- **MLP vs Conv1D, single-repetition test:** E3 vs E4: R@5 0.1165 vs 0.1563, R@10 0.200 vs 0.259, R@1 0.032 vs 0.038. The direction favours Conv1D.
- **AVG vs Non-AVG training, averaged test:** AVG-trained models outperform Non-AVG-trained ones on averaged test (R@5 0.38/0.41 vs 0.255/0.215). On Single-repetition test, Non-AVG-trained models are marginally better (Conv1D 0.156 vs 0.125; MLP 0.117 vs 0.112 R@5).

---

## 8. Discussion


 1. **QR1: Above chance**. All four models exceed chance on both test representations, with differences that are statistically well beyond binomial chance. For single-repetition EEG, Recall@1 is only 2.8–3.8% (about 6–8 times chance, but absolute performance is low), and Recall@5 11–16%. For 80-averaged EEG, Recall@5 reaches 0.22–0.41. This shows the learned map aligns EEG with CLIP embeddings *better than chance* for this subject and these 200 test images.

2. **QR2: Average vs Non-Average.** 
The results suggest that retrieval performance depends on the consistency of repetition averaging between training and test data. Models trained on averaged EEG performed better when evaluated on averaged test data, whereas models trained on non-averaged EEG performed better on non-averaged test data. This indicates that training with non-averaged repetitions is beneficial under a matching non-averaged evaluation setting, rather than providing a general performance advantage. Therefore, the current results do not establish that averaging universally improves retrieval; instead, they highlight the importance of matching the EEG representation across training and evaluation.

3. **RQ3: MLP vs Conv1D.** Across all eight experimental settings, the Conv1D models consistently achieved better test retrieval performance than the MLP models. Notably, the fully connected layers of the Conv1D architecture were matched to those of the MLP, making the feature extraction stages the primary architectural difference. This result suggests that convolutional layers provide an effective way to extract and represent EEG features for the retrieval task, offering an advantage over directly processing flattened EEG signals with fully connected layers. Additionally, three MLP configurations were evaluated on the validation set: a three-layer linear architecture, a two-layer linear architecture, and a two-layer linear architecture with dropout. Among these configurations, the MLP with two linear layers and dropout achieved the best validation performance. This suggests that simplifying the network and introducing dropout may help improve generalization by reducing overfitting.

---

## 9. Conclusion 

For a single subject, EEG encoders trained with symmetric InfoNCE against pre-computed CLIP image embeddings successfully retrieved held-out test images at rates well above chance. Recall@5 ranged from 0.22 to 0.41 with EEG averaged across 80 repetitions and from 0.11 to 0.16 with single-repetition EEG, compared with a chance level of 0.025 in a 200-image gallery. Test-time averaging was the most clearly demonstrated factor associated with improved retrieval performance. These results establish an initial baseline for EEG-to-image retrieval and provide a foundation for future work aimed at improving model architectures, training strategies, and retrieval performance.

