# Implementation Plan: 3D Facial Reconstruction for Sign Language Video

## TokenFace-style tokenized ViT encoder + SMIRK analysis-by-neural-synthesis training + score-biased temporal transformer

---

## 1. Overview

Goal: accurate FLAME-based 3D face reconstruction on sign language video, with faithful subtle/extreme expressions (mouthings, brow movements, blinks) and robustness to hand-over-face occlusions.

Approach:
- **Encoder architecture** from TokenFace (Zhang et al., ICCV 2023): a ViT with learnable per-parameter component tokens appended to image tokens.
- **Training supervision** from SMIRK (Retsinas et al., CVPR 2024): neural rendering (UNet image-to-image translator) reconstruction path + augmented expression cycle path, extended with direct 3D vertex supervision on FLAME-registered scan data (TokenFace-style hybrid 2D/3D training).
- **Temporal model**: a small transformer over per-frame component tokens with ALiBi-style attention biases driven by (a) temporal distance and (b) a precomputed per-frame face-visibility score, so occluded/blurred frames attend to reliable neighbors.

The SMIRK codebase is available and should be reused wherever possible (differentiable rasterizer, masking, pixel transfer, expression templates, emotion network, UNet architecture, loss implementations). Do not reimplement SMIRK components from scratch.

---

## 2. Architecture

Five trainable component groups.

### 2.1 Spatial ViT (SViT)
- ViT-B/16, hidden dim **768**.
- Initialized from **FaRL-B** pretrained weights (as in TokenFace).
- Learned absolute position embeddings, initialized from FaRL, **trainable** (interpolate the grid if input resolution differs from FaRL's 224×224 pretraining).
- Input per frame: image patch tokens + 4 learnable component tokens (Sec 2.2), all 768-dim.
- Output: the 4 component tokens after the final transformer layer (image tokens discarded).

### 2.2 Learnable component tokens
4 tokens, dim 768, **initialized to zero** (as in TokenFace). Each maps to a FLAME/camera parameter group via its own MLP head:

| Token | Target parameters | Dim |
|---|---|---|
| Shape | FLAME shape β | 300 |
| Expression | FLAME expression ψ + 2 eyelid blendshapes | 100 + 2 |
| Jaw | jaw pose θjaw | 3 |
| Camera + global rotation | 1 scale + 3 global rotation + 2D translation (weak-perspective/orthographic, matching SMIRK's reused renderer - no depth/tz term) | 6 |

- Eyelid blendshapes: as in SMIRK, values in [0, 1] (0 = open, 1 = closed). Implement the constraint with a **sigmoid** on the MLP output (not a hard clamp — clamping kills gradients at the boundary).
- No texture or lighting tokens: the neural renderer receives sparsely sampled input pixels and infers appearance itself (SMIRK design).

### 2.3 Output MLP heads
- **One MLP head per token** (4 heads), trained from scratch, mapping 768 → parameter dim per the table above.

### 2.4 Temporal Transformer (TT)
- 3 layers, **8 heads** (768 / 8 = 96 per head), dim 768, trained from scratch.
- **Residual delta design**: TT output layer is zero-initialized so that at init the TT is an identity map over the SViT tokens (output delta = 0). TT refines tokens; it does not replace them.
- Operates on the 4 component tokens per frame across a temporal window: **full joint attention over all 4 tokens × w frames, no intra-frame masking** (same-frame cross-component attention is kept; cross-time and cross-component-cross-time links are the useful ones, and same-frame neighbors help the residual delta calibrate itself).
- Attention bias (see Sec 4) replaces positional embeddings entirely — no learned/absolute position embeddings, to allow generalization across sequence lengths.

### 2.5 Neural rendering UNet
- SMIRK's image-to-image translator, trained from scratch. Use the SMIRK implementation directly (UNet with encoder/decoder shortcut connections + residual blocks; the shortcuts are required for gradient flow to the encoder).
- Input: monochrome rasterization of the predicted mesh ⊕ masked input image with ~1% randomly retained face pixels (SMIRK masking; keep the 1% ratio — SMIRK's ablation shows 5% breaks expression control).

---

## 3. Inference flow

**Video:**
1. Per frame: patchify → add position embeddings → image tokens; append the 4 learned component tokens; run SViT → 4 output tokens per frame.
2. TT: input all frames' tokens + per-frame visibility score (Sec 4); windowed biased attention → refined tokens per frame.
3. Per-token MLP heads → FLAME + camera parameters per frame → FLAME → mesh.

**Single image:** skip step 2 (SViT → MLP heads directly).

---

## 4. Temporal attention with visibility-score bias

### 4.1 Face-visibility score (offline preprocessing)
- Per frame: score = (area of visible face pixels from an occlusion-robust face segmentation model, e.g. XSeg) / (area of the RetinaFace-detected face bounding box). This score calculation is implemented in `scripts/visible_face_ratio.py`. Range [0, 1]; higher = less occluded = more trustworthy SViT tokens.
- **Precomputed offline for every video frame in every dataset** and stored alongside the data. The identical computation is used at inference time.

### 4.2 Window
- Attention restricted to a **centred** window of **w = 15** frames (config parameter): query frame i attends only to frames `[i - w//2, i + w//2]`.
- Rationale recorded: smoothness needs ~3 adjacent frames; occlusion infill needs longer reach. w was raised from an initial 11 to 15 to give more reach for the occlusion-span risk below.
- Implemented as true local (sliding-window) attention (`model/temporal.py`), not a mask on top of dense attention - a mask alone would still cost O(N²) regardless of w (`nn.MultiheadAttention` always computes the full QK^T; masking only changes what survives softmax). True local attention costs O(N·w), linear in however many frames N it's called with. This matters for two reasons:
  1. **Avoid compute blow-up on long videos at inference.** A full-length inference video can be run through TT in one pass without cost exploding.
  2. **Avoid boundary artifacts from chunking long videos.** Because cost no longer depends on N, inference never needs to chop a long video into fixed-size clips for compute reasons - eliminating the artificial "seam" every clip-length frames where a frame would otherwise lose access to real neighbors purely because of where a chunk boundary fell. Boundary effects now only occur at the true start/end of the video (unavoidable - there simply aren't w//2 neighbors there), not at arbitrary chunk seams.
- Training clip length (`max_frames` in `dataset_processing/config/dataloader.yaml`) remains a separate, larger batching decision, unrelated to w now that attention cost no longer depends on clip length.
- ⚠ Open risk: signing occlusions can exceed w frames. Before finalizing, measure the occlusion-span distribution on the sign language datasets using the visibility score, and increase w further if a large fraction of occlusion spans lack clean frames within the window. Keep w a config knob.

### 4.3 Score normalization
- Within each window, subtract the window mean from each frame's score → normalized relative scores. (Mean subtraction only — do **not** z-score/divide by std: unit-variance rescaling would amplify negligible score noise in near-uniform windows into large attention biases. Mean subtraction preserves variation magnitude, so only genuinely low-visibility frames receive a strong bias.)

### 4.4 Attention bias (ALiBi-style)
For query frame i, key frame j, head h, add to the pre-softmax QK logits:

```
bias(i, j, h) = m_h * s̃_j − n_h * |i − j|
```

- `s̃_j` = normalized visibility score of the **key** frame (attend more to reliable frames).
- `|i − j|` = temporal distance, penalized (ALiBi-style, negative slope).
- `m_h`, `n_h`: **fixed** per-head constants (not learned). Heads span a genuine **grid** of (m, n) combinations rather than a single m = k·n line: 4 m-values × 2 n-values = 8 heads (one (m, n) pair per head, all combinations covered). Both are geometric series:
  - `m ∈ {6.25, 12.5, 25, 50}` — spans up to ~50 so that an extreme score deviation (s̃ ≈ 0.1) can dominate over a typical distance penalty even at the grid's low end (with mean-subtracted scores in roughly [−0.1, 0.1] and distances in [−5, 5]).
  - `n ∈ {0.5, 0.25}` — ALiBi's own first two slopes from its standard 8-head geometric sequence (2⁻¹, 2⁻²; see Press et al., "Train Short, Test Long").
- All 4 component tokens of frame j receive the same frame-level score; attention is full joint attention over all 4 tokens × w frames within the window (component tokens carry component-type embeddings so the TT can distinguish them; temporal order is conveyed solely via the distance bias).

---

## 5. Data

### 5.1 Datasets

| Type | Datasets | Losses |
|---|---|---|
| 2D image | CelebA, FFHQ, BUPT-Balancedface, NoW-excluded misc | self-supervised (SMIRK) |
| 2D video | MEAD, AFEW-VA | self-supervised + temporal |
| 3D image (image + registered mesh pairs) | LYHM, DAD-3DHeads, NoW **excluded from training** | direct vertex supervision |
| 3D video (4D sequences) | FaMoS, CoMA, VOCASET | vertex supervision + temporal |
| 2D sign language video | How2Sign, CSL-Daily, PHOENIX-2014T | self-supervised + temporal |

- **NoW is never trained on** (reserved benchmark).
- Sign language datasets use their **official train/dev/test splits**; all other datasets are train-only. (Already prepared by the user.)
- Identity labels available (for identity-swap losses): FaMoS, Headspace, CoMA, VOCASET, BUPT-Balancedface, MEAD, AFEW-VA, CSL-Daily, PHOENIX-2014T, How2Sign.
- 3D datasets are stored as (2D image, FLAME-registered 3D mesh) pairs. All listed 3D datasets are already in FLAME topology — no registration/conversion work required.

### 5.2 Batching
- **Homogeneous per-category batches, combined jointly every step** — each training step draws one batch from *every* category simultaneously (via a shared combined loader; the smaller categories cycle/repeat to keep pace with the largest), rather than stochastically picking a single category per step. Relative influence between data types is controlled by **loss-weight hyperparameters** (Sec 6), not by sampling probability — kept as tunable config, no fixed ratio prescribed here. This also means the epoch length is set by the largest category, so smaller-but-valuable categories (3D data is scarce) are seen proportionally *more* often per epoch, not less.
  - Within the 2D pool, expression-rich data (MEAD frames, sign language frames) is up-weighted relative to plain face datasets (CelebA/FFHQ/BUPT) - exact weighting is a tunable config knob.
  - Temporal pass category composition (sign language / other 2D video / 3D video) is likewise a tunable per-category batch-size/weight config, not a fixed prescribed split.
- **Video datasets switch between two modes depending on whether TT is being trained in the current pass:**
  - **TT frozen** (Stage 1; Stage 2 Pass A/B): video datasets contribute **randomly sampled individual frames** (as in SMIRK) — one fresh random frame per access, so a batch of "2D-loss" or "3D-loss" samples can mix real images and single video frames interchangeably. Re-sampled every access (not fixed once per video), so a long training run eventually covers most of a video's frames, not just one.
  - **TT training** (Stage 2 Pass C, the temporal pass): video datasets contribute **full clips** (all frames in the window), since TT needs a genuine temporal sequence to operate on.

### 5.3 Other preprocessing
- 2D landmarks: as in SMIRK — MediaPipe (face interior, 92 pts) + FAN (16 boundary pts), precomputed.
- MICA shape predictions precomputed per identity/image for the MICA distillation loss.
- Visibility scores per video frame (Sec 4.1).
- FLAME expression templates for the cycle path: reuse SMIRK's FaMoS-fitted templates.

---

## 6. Losses

Notation: I input image, I′ = UNet output, M predicted mesh, V vertices.

| Loss | Definition | Applied to |
|---|---|---|
| Photometric | L1(I′, I) | 2D recon pass |
| VGG | L1 of VGG features of I′ vs I | 2D recon pass |
| Landmark | L2 of projected 3D landmarks vs detected 2D landmarks; includes eye-closure and mouth/lip-closure terms (SMIRK/DECA-style) | pretraining + recon pass |
| MICA shape distillation | L2 between predicted β and MICA's predicted β | pretraining + recon pass |
| Mesh (3D) | region-weighted L1 between predicted and GT vertices. Per-vertex weight from FLAME_masks.pkl regions: 2.0 for expressive regions (lips, eye_region, left/right_eye_region, nose, forehead), 1.0 for face (general skin) and boundary, 0.0 for left/right_eyeball (FLAME's eye joints are fixed, not predicted, per Sec 2 - no mechanism to fit these) and everything not in `face` (neck, ears, scalp - outside the face-only render region, Sec 2.5/9) | 3D batches, both stages |
| Vertex consistency Lvc (3D identity swap) | swap β between two same-identity samples; L1 between resulting vertices and GT (TokenFace Eq. 5) | 3D identity-labeled batches |
| Emotion | L2 between pretrained emotion-net features of I′ and I; **UNet frozen for this loss** (only the expression pathway updates) | recon pass |
| Expression cycle consistency | augment ψ (permutation / perturbation / template injection / zeroing, with jaw+eyelid co-augmentation), render via UNet with **pixel transfer**, re-encode; L2(ψ̂, ψaug) (SMIRK Eq. 2) | augmentation pass |
| Identity (β) cycle consistency | same cycle; L2 between re-encoded β and original β. Applied on **both** alternations (encoder update and UNet update) — deviation from SMIRK (where Eβ is frozen); acts as an additional encoder disentanglement signal here. Note: self-consistency only — MICA + Lvc remain the accuracy anchors. | augmentation pass |
| Temporal smoothness | **Velocity** penalty (L1, mean absolute first difference, \|p(t) − p(t+1)\|) applied uniformly to expression (+eyelids), jaw, camera+global-rotation, and shape parameters — discourages frame-to-frame jumps while tolerating smooth, sustained motion. (Originally an acceleration/L2 formulation on expression/jaw/camera; switched to velocity/L1 after Stage 2 training was observed to collapse to a near-fixed mesh — the L2 acceleration term's quadratic growth on real motion over-suppressed genuine expressiveness.) | temporal pass |
| Regularization | L2 on expression parameters (and standard FLAME param regularizers - shape, jaw) | all passes |

Starting loss weights (from SMIRK): cycle 10, landmark 100, VGG 10, photometric 1, emotion 1. TokenFace anchors: mesh λmesh = 2.0, Lvc λvc = 1.2, 2D/3D balance 0.4/0.6, regularization 1e-4 (uniform across shape/expression/jaw - TokenFace uses one weight for all FLAME params, unlike SMIRK's differentiated per-parameter weights). For losses with no published anchor, starting guesses (tune these first on the sign language dev splits): MICA shape distillation 1.0; β identity cycle consistency 10 (mirroring the expression cycle weight, per SMIRK's "similar to Eq. 2"); temporal smoothness — velocity term 0.1 (lowered from an original 1.0 after the fixed-mesh collapse noted above; keep smoothness weight low initially and raise only if jitter persists — over-weighting damps mouthings). All weights are config knobs.

---

## 7. Training stages

### Stage 1 — Pre-training (stabilize SViT + MLP heads before UNet/TT)
- **Updating:** component tokens, SViT (incl. position embeddings), MLP heads.
- **Frozen:** UNet, TT (neither participates).
- **Data:** image-level only (videos as frame pools).
- **Losses:**
  - 2D batches: landmark losses + MICA shape distillation
  - 3D batches: mesh loss + Lvc
  - Output regularization on MLP outputs
- Purpose (per SMIRK): the UNet must later train against an encoder whose rendered geometry is already meaningful, otherwise it learns to ignore the geometry input.
- Reference schedule: SMIRK pretrains 60k iterations, Adam, lr 5e-4.

### Stage 2 — SMIRK training + temporal training (three alternating passes)

Cycle through the three passes each iteration (or in a fixed pattern; make the pattern a config knob).

**Pass A — Reconstruction pass** (image-level batches; this includes randomly sampled individual frames from all video datasets, treated as images)
- 2D batches: photometric + VGG + landmark + emotion + MICA + regularization, through the full SMIRK reconstruction path (mask → sample 1% pixels → UNet → losses).
- 3D batches: mesh + Lvc.
- **Updating:** tokens, SViT, MLP heads, UNet. **Frozen:** TT. UNet additionally frozen w.r.t. the emotion loss only.

**Pass B — Augmentation (cycle) pass** (2D images only; 3D datasets contribute their 2D images, meshes ignored)
- Expression augmentation (permutation, perturbation, template injection, zero-expression) + **pixel transfer** → UNet render → re-encode → expression cycle loss + β identity cycle loss + regularization.
- **Updating:** alternate between (tokens + SViT + MLP) and (UNet) — SMIRK's alternating freeze, preventing the UNet from compensating for encoder errors. **Frozen:** TT.

**Pass C — Temporal pass** (video batches: 2D video, sign language video, 3D video)
- Full pipeline SViT → TT → MLP heads on clips; SViT tokens cannot be precomputed since SViT is still training in passes A/B.
- Losses: per-frame reconstruction losses (photometric/VGG/landmark for 2D video; mesh for 3D video) + temporal smoothness + regularization.
- **Updating:** TT only. **Frozen:** SViT, tokens, MLP heads, UNet.
- Clip length ≥ the inference window w; train at the window size intended for inference.

- Reference schedule: SMIRK's core phase is 250k iterations, lr 1e-3 with cosine annealing restarted per epoch, batch 32.

**Out of scope for now** (planned later experiments): synthetic occlusion training, dropped-frame (masked-token) training, learned score embeddings, backbone comparison (MARLIN/DINOv2), end-to-end SViT+TT fine-tuning.

---

## 8. Evaluation

Use `run_evaluation_dataset.py` as the reference implementation for the evaluation pipeline.

## 9. Implementation notes

- Reuse from SMIRK repo: rasterizer + FLAME wrapper (incl. eyelid blendshapes), masking + pixel sampling, pixel transfer, expression templates, emotion network, UNet, loss functions, training loop skeleton.
- New components to implement: tokenized ViT encoder (FaRL init + 4 component tokens + per-token heads), TT with score/distance-biased windowed attention, visibility-score preprocessing pipeline, homogeneous multi-type batch sampler, three-pass stage-2 scheduler with the freezing map above.
- Precompute and cache: landmarks, MICA shapes, visibility scores, face crops.
- Checkpointing per stage, stored in `$PROJECTDIR`; stage 2 resumes from stage 1.