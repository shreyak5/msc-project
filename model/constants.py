"""Constants for the tokenized ViT encoder and FLAME parameterization (implementation-plan.md Sec 2)."""

# FLAME parameter dimensions
FLAME_SHAPE_DIM = 300
FLAME_EXPRESSION_DIM = 100
NUM_EYELID_PARAMS = 2
FLAME_JAW_POSE_DIM = 3

# Camera + global rotation (Sec 2.2): weak-perspective/orthographic camera, matching
# SMIRK's reused renderer exactly (renderer/util.py's batch_orth_proj expects
# [scale, tx, ty] - no depth/tz term; SMIRK's own PoseEncoder/FLAME split is the same
# 1 scale + 2D translation + 3D FLAME global rotation, just produced by two separate
# sub-networks there instead of one token). No tz: for a small, roughly fronto-parallel
# object like a cropped face, depth-translation and scale are largely redundant/
# degenerate from monocular RGB alone (moving closer vs. scaling up look the same),
# so the standard convention in this line of work (DECA/EMOCA/SMIRK) keeps only scale.
CAMERA_SCALE_DIM = 1
GLOBAL_ROTATION_DIM = 3
TRANSLATION_DIM = 2

# Component token parameter-group dims (output of each per-token MLP head, Sec 2.2/2.3)
SHAPE_TOKEN_DIM = FLAME_SHAPE_DIM
EXPRESSION_TOKEN_DIM = FLAME_EXPRESSION_DIM + NUM_EYELID_PARAMS
JAW_TOKEN_DIM = FLAME_JAW_POSE_DIM
CAMERA_TOKEN_DIM = CAMERA_SCALE_DIM + GLOBAL_ROTATION_DIM + TRANSLATION_DIM

# Camera token layout: [scale(1), global_rotation_axis_angle(3), translation_xy(2)],
# in that order. global_rotation feeds FLAME's kinematic root joint (model/flame.py);
# scale + translation_xy feed the renderer's weak-perspective projection - FLAME
# itself never sees scale/translation.
CAMERA_SCALE_SLICE = slice(0, CAMERA_SCALE_DIM)
CAMERA_ROTATION_SLICE = slice(CAMERA_SCALE_DIM, CAMERA_SCALE_DIM + GLOBAL_ROTATION_DIM)
CAMERA_TRANSLATION_SLICE = slice(CAMERA_SCALE_DIM + GLOBAL_ROTATION_DIM, CAMERA_TOKEN_DIM)

NUM_COMPONENT_TOKENS = 4

# SViT backbone (ViT-B/16, Sec 2.1)
SVIT_IMG_SIZE = 224
SVIT_PATCH_SIZE = 16
SVIT_EMBED_DIM = 768
SVIT_DEPTH = 12
SVIT_NUM_HEADS = 12
SVIT_MLP_RATIO = 4.0

# Temporal Transformer (Sec 2.4)
TT_EMBED_DIM = 768
TT_DEPTH = 3
TT_NUM_HEADS = 8
# Not specified by the plan (only dim/depth/heads are); using the same standard
# ViT-style expansion ratio as SVIT_MLP_RATIO.
TT_MLP_RATIO = 4.0

# Attention bias window (Sec 4.2): a centred window (radius = TT_WINDOW_SIZE // 2 on
# each side of the query frame), enforced as true local attention in model/temporal.py
# (not just a mask on top of dense attention) so compute/memory scale with N*w, not N^2.
TT_WINDOW_SIZE = 15

# Sec 4.4 (revised): visibility and distance are no longer combined into one shared
# per-head bias. Instead, TT_NUM_HEADS splits into a dedicated visibility-only head
# (attention weights driven purely by raw per-frame visibility, no QK/distance term)
# and the remaining heads, which keep ordinary QK^T content attention plus a
# distance-only ALiBi bias (no visibility term). See model/temporal.py.
TT_NUM_VISIBILITY_HEADS = 1


def _alibi_slopes(num_heads: int) -> tuple[float, ...]:
    """Press et al.'s standard ALiBi geometric slope sequence ("Train Short, Test
    Long"): slope_i = 2^(-8*i/num_heads) for i = 1..num_heads. Sized to however many
    QK+ALiBi (distance-only) heads TT actually has, not hardcoded to a specific count."""
    return tuple(2.0 ** (-8.0 * i / num_heads) for i in range(1, num_heads + 1))


# ALiBi distance-only slopes for the QK+ALiBi heads (all TT_NUM_HEADS except the
# TT_NUM_VISIBILITY_HEADS dedicated visibility head): bias(i,j,h) = -n_h * |i-j|.
TT_ALIBI_SLOPES: tuple[float, ...] = _alibi_slopes(TT_NUM_HEADS - TT_NUM_VISIBILITY_HEADS)

# Alternate TT design (SimpleTT, model/temporal.py): no visibility-only head at all -
# every one of TT_NUM_HEADS heads is an ordinary QK+ALiBi head, so this needs its own
# full-width slope sequence rather than reusing TT_ALIBI_SLOPES (sized to 7, not 8).
SIMPLE_TT_ALIBI_SLOPES: tuple[float, ...] = _alibi_slopes(TT_NUM_HEADS)

# GatedTT (model/temporal.py): default sharpness for the post-softmax
# visibility-gating exponent (gate_j = visibility_j ** gamma). gamma=1 leaves the
# raw score as-is; gamma>1 sharpens (pushes low-visibility positions down harder);
# gamma->0 approaches a no-op (gate -> 1 everywhere, since x**0 == 1).
TT_GATE_GAMMA = 3.0

# Softmax temperature for the visibility-only head: weight(i,j) = softmax_j(visibility_j
# / TT_VISIBILITY_TEMPERATURE). 1.0 would leave the raw visibility score unscaled;
# <1 sharpens the distribution toward the single most-visible frame in the window
# (0.1 chosen to make that effect pronounced), >1 flattens it toward uniform.
TT_VISIBILITY_TEMPERATURE = 0.01

# Neural rendering UNet (Sec 2.5). Matches SMIRK's actual trainer instantiation
# (smirk_trainer.py: SmirkGenerator(in_channels=6, out_channels=3, init_features=32,
# res_blocks=5)), not SmirkGenerator's own class defaults (3/1/16/3). Input is the
# renderer's shaded mesh image (3 channels, visually grayscale - uniform albedo x
# shading - but stored 3-channel) concatenated with the masked 3-channel RGB image
# (~1% of real face pixels retained, Sec 2.5); output is the full RGB reconstruction.
UNET_IN_CHANNELS = 6
UNET_OUT_CHANNELS = 3
UNET_INIT_FEATURES = 32
UNET_RES_BLOCKS = 5

# FLAME model assets (model/flame/flame.py), paths relative to the repo root.
FLAME_MODEL_PATH = "assets/FLAME2020/generic_model.pkl"
FLAME_LMK_EMBEDDING_PATH = "assets/landmark_embedding.npy"
FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH = "assets/mediapipe_landmark_embedding/mediapipe_landmark_embedding.npz"
FLAME_L_EYELID_PATH = "assets/l_eyelid.npy"
FLAME_R_EYELID_PATH = "assets/r_eyelid.npy"
EXPECTED_NUM_FLAME_VERTICES = 5023

# Renderer assets (model/flame/renderer.py), paths relative to the repo root. No
# separate head-template mesh asset needed: verified empirically that
# assets/head_template.obj's face connectivity is identical to FLAME's own
# faces_tensor (model/flame/flame.py), so the renderer takes that directly rather
# than loading a redundant duplicate. FLAME_masks.pkl is a genuinely separate,
# curated vertex-region annotation (face/neck/ears/scalp/...), not derivable from
# the FLAME model file itself.
RENDERER_FLAME_MASKS_PATH = "assets/FLAME_masks/FLAME_masks.pkl"
# Sec 2.5/Sec 9: render only the face region, not the full head - matches SMIRK's own
# config (config_train.yaml: render.full_head: False).
RENDERER_FULL_HEAD = False
RENDERER_IMAGE_SIZE = SVIT_IMG_SIZE

# Masking / pixel-transfer (model/flame/masking.py, Sec 2.5: "masked input image with
# ~1% randomly retained face pixels"). mask_ratio matches SMIRK's actual trainer
# config (config_train.yaml: mask_ratio: 0.01). FLAME_masks_triangles is a curated
# map of FLAME-region-name -> triangle indices, used to bias which triangles are
# eligible to be sampled as retained pixels (Sec 2.5's ablation note: keep the 1%
# ratio - 5% breaks expression control).
MASK_RATIO = 0.01
# 0 = no dilation - SMIRK's own config uses 10 (config_train.yaml:
# mask_dilation_radius), a safety margin against imprecision in their own
# convex-hull-of-landmarks mask. This project's face_mask comes from XSeg
# segmentation instead (dataset_processing's face-parsing cache), a
# meaningfully more precise per-pixel mask than a landmark-hull approximation
# - judged accurate enough not to need that safety margin.
MASK_DILATION_RADIUS = 0
FLAME_MASKS_TRIANGLES_PATH = "assets/FLAME_masks/FLAME_masks_triangles.npy"
NUM_FLAME_FACES = 9976
# Per-triangle-region sampling weight (0 = never sample, e.g. neck/ears/eyeballs
# should never leak into the sparse appearance hint; 0.5 = half-weight for
# lips/nose, since sampling those could leak mouth-shape/expression information
# directly, partially defeating the point of forcing geometry-only expression
# inference; 1.0 = full weight for "clean" skin regions).
FLAME_MASK_AREA_WEIGHTS: dict[str, float] = {
    "neck": 0.0,
    "right_eyeball": 0.0,
    "right_ear": 0.0,
    "lips": 0.5,
    "nose": 0.5,
    "left_ear": 0.0,
    "eye_region": 1.0,
    "forehead": 1.0,
    "left_eye_region": 1.0,
    "right_eye_region": 1.0,
    "face_clean": 1.0,
    "cleaner_lips": 1.0,
}

# Regularization losses (model/losses/regularization.py, Sec 6, "all passes"): L2
# penalty pulling params toward zero (FLAME's shape/expression bases are zero-
# centered PCA coefficients - zero already means "neutral"/"mean face"). Kept as
# separate per-parameter-group constants (so each can be tuned independently later).
# Originally all equal (1e-4, TokenFace's uniform value) - REG_EXPRESSION_WEIGHT
# split out and raised to 1e-3 after measuring real per-group l2_regularization
# magnitudes on a held-out test image (a 100-dim checkpoint, but l2_regularization
# is a MEAN not a sum, so this isn't a 50-vs-100-dim artifact): expression's raw
# mean(x^2) was 16.4, vs shape's 0.31 and jaw's 0.0013 - expression alone already
# accounts for ~98% of the combined "reg" metric regardless of these weights, so
# shape/jaw were left untouched (raising them wouldn't touch this at all) while
# expression was targeted specifically. SMIRK's own literal expression_regularization
# (1e-3, config_train.yaml) was checked against real Stage 2 loss magnitudes and
# found to still be a rounding error (~2.8% of Pass A's 2D total, ~1.2% of Pass B's
# expr_cycle - Pass B's cycle-consistency loss has no SMIRK equivalent to calibrate
# against). Tried 1e-2 (100x the original uniform value) first: ~22% of Pass A's 2D
# total (comparable to VGG's own share), ~11% of Pass B's expr_cycle - confirmed
# working (expression norm on a held-out test image dropped from ~28.6 to ~9.1 at
# matched step 12999), but judged too strong after watching it continue training
# further (visibly over-suppressing expressiveness). Halved to 5e-3 (~13% of Pass
# A's 2D total, ~6% of Pass B's expr_cycle) - visually confirmed natural and
# slightly more expressive than 1e-2. Lowered further to SMIRK's own literal
# 1e-3 (~2.8%/~1.2%, originally judged "too weak to matter" from the numbers
# alone) per explicit request to push toward more expressive outputs, having
# already seen 5e-3 trend the right direction - an empirical call favoring
# more expressiveness over the theoretical share-of-loss argument. Set back to
# 5e-3 for a one-off experiment (stage2_50_AB_unet100.yaml only, "_v5"
# checkpoint_dir) combining it with the new IDENTITY_CYCLE_LOSS_WEIGHT/
# EMOTION_LOSS_WEIGHT changes below, rather than testing those two against
# 1e-3 - the sibling _v4 lines (currently checkpointed at 1e-3) are NOT
# affected by this yet since they're not currently running; this constant is
# global, so resuming any of them without flipping this back first would
# train them at 5e-3 instead of their own established 1e-3.
REG_EXPRESSION_WEIGHT = 5e-3
REG_JAW_WEIGHT = 1e-4
REG_SHAPE_WEIGHT = 1e-4

# Camera scale regularization (model/losses/regularization.py's
# log_scale_regularization) - a later, separate addition, not part of the
# uniform-1e-4 group above (see that function's own docstring for the log-
# ratio-toward-a-reference rationale, distinct from plain L2-toward-zero).
# CAMERA_SCALE_REFERENCE=7.0 matches both SMIRK's own hand-chosen camera-
# scale init constant (src/smirk_encoder.py's PoseEncoder) and this project's
# own Stage 1 pretrain endpoint (measured ~7.4, same weak-perspective
# convention). REG_CAMERA_SCALE_WEIGHT=0.1 - deliberately NOT matched to the
# 1e-4 group: measured log(scale/7)^2 at real observed collapsed scale values
# (~2.7-4.4 across Stage 2 runs) is order 0.1-0.9, and landmark/closure/etc.
# (the terms currently outvoting camera scale with no anchor at all) are
# order 0.01-0.5 raw - 1e-4 would make this term contribute <1e-4 to the
# total loss, negligible next to those, i.e. present in code but not in
# practice. 0.1 is a starting guess (100x the uniform group) chosen to be
# competitive with those terms instead; revisit empirically.
CAMERA_SCALE_REFERENCE = 7.0
REG_CAMERA_SCALE_WEIGHT = 0.1

# Mesh (3D) region-weighted vertex loss (model/losses/mesh.py, Sec 6). Per-vertex
# weight built from FLAME_masks.pkl regions (RENDERER_FLAME_MASKS_PATH - same asset
# already used by the renderer). Not derived from SMIRK/TokenFace's own code (SMIRK
# doesn't train with direct 3D mesh supervision at all; TokenFace's exact scheme
# isn't published) - a reasoned default: up-weight the most expressive regions,
# zero out regions with no fitting mechanism (eyeballs - FLAME's eye joints are
# fixed, not predicted, Sec 2) or outside the face-only render region (Sec 2.5/9).
MESH_LOSS_EXPRESSIVE_WEIGHT = 2.0
MESH_LOSS_FACE_WEIGHT = 1.0
MESH_LOSS_BOUNDARY_WEIGHT = 1.0
MESH_LOSS_EYEBALL_WEIGHT = 0.0
MESH_LOSS_DEFAULT_WEIGHT = 0.0  # neck/ears/scalp - never explicitly set, stay at this
MESH_LOSS_EXPRESSIVE_REGIONS: tuple[str, ...] = (
    "lips",
    "eye_region",
    "left_eye_region",
    "right_eye_region",
    "nose",
    "forehead",
)
MESH_LOSS_EYEBALL_REGIONS: tuple[str, ...] = ("left_eyeball", "right_eyeball")

# Overall loss coefficients (TokenFace anchors, Sec 6 weights paragraph: "mesh
# λmesh = 2.0, Lvc λvc = 1.2") - these scale the mesh/Lvc losses' contribution to
# the total training objective, distinct from the per-region weights above (which
# only shape the region-weighted average *within* the mesh loss itself). Direct 3D
# ground-truth supervision (dad_3dheads/coma/vocaset/famos/headspace) is a harder,
# less gameable anchor than the 2D proxies (landmark/VGG/photometric) - doubled
# from TokenFace's 2.0 to 4.0 specifically because raising LANDMARK_LOSS_WEIGHT to
# 100 (above) roughly doubled Pass A's 2D total, which - under the fixed 0.4/0.6
# LOSS_BALANCE_2D/3D split - passively diluted mesh's share of Pass A's combined
# total from ~28% to ~19% even though mesh's own value didn't change (measured
# same Pass A window, steps 12750-13800 of stage2_50_AAB_UNet100). Doubling here
# restores mesh to ~32% of the combined total - back above its original share,
# not just compensating for the dilution. Within the 3D loss itself mesh already
# dominates lvc/reg_3d (~83%/15%/1.5% respectively at the old 2.0/1.2 lambdas) -
# lvc/reg_3d were left untouched since they weren't the term being diluted.
MESH_LOSS_LAMBDA = 4.0
VERTEX_CONSISTENCY_LOSS_LAMBDA = 1.2

# Landmark loss (model/losses/landmark.py, Sec 6). Sec 6 weights paragraph:
# "Starting loss weights (from SMIRK): ... landmark 100 ...". SMIRK's own code
# applies this weight to both the FAN and MediaPipe terms independently, then
# sums them (not averaged) - matched here. Originally lowered from SMIRK's
# literal 100 to 10 (compared only against the losses actually co-active
# during Stage 1 pretraining - mesh 2.0, Lvc 1.2, MICA 1.0 - literal 100 would
# have been a 50-100x outlier there). Restored to SMIRK's literal 100 after
# measuring real Stage 2 Pass A loss magnitudes (steps 12750-13800 of
# stage2_50_AAB_UNet100): at weight 10, landmark's weighted contribution
# (~0.027) was only ~8% of the 2D loss total (~0.325), well below vgg/mica/
# closure - visually correlated with SMIRK-comparison reconstructions looking
# more exaggerated/under-constrained than SMIRK's own. At 100, landmark lands
# at ~47% of the 2D total - the same order of magnitude as everything else
# combined, not a 50-100x outlier, given this project's current (already-
# lowered-from-SMIRK) VGG/MICA weights.
LANDMARK_LOSS_WEIGHT = 100.0
# Eye/lip closure terms have no SMIRK precedent at all (an EMOCA-derived
# addition, Sec 6d-3). Was previously tied to LANDMARK_LOSS_WEIGHT ("reusing
# the same weight to start rather than introducing a second untuned constant")
# - split into its own constant when LANDMARK_LOSS_WEIGHT was raised to 100,
# since closure's raw magnitude is already ~3.6x landmark's (measured same
# Pass A window), so scaling both together would have made CLOSURE the
# dominant 2D term (~68% of the total) rather than fixing landmark's
# under-weighting as intended. Kept at the prior shared value (10) - untouched
# by this change, revisit independently if closure itself looks off.
CLOSURE_LOSS_WEIGHT = 10.0

# 2D/3D batch balance (Sec 6 weights paragraph: "2D/3D balance 0.4/0.6") - Stage 1
# (and Pass A) draw one 2D batch and one 3D batch every step (Sec 5.2's joint
# per-category batching); these scale each batch type's total loss before summing
# into the single combined backward pass.
LOSS_BALANCE_2D = 0.4
LOSS_BALANCE_3D = 0.6

# Expression cycle consistency augmentation (model/losses/cycle.py, Sec 6). SMIRK's
# own precomputed FaMoS-fitted expression templates (direct iterative FLAME fitting
# on FaMoS's extreme/asymmetric expressions), used for the "template injection"
# augmentation type. Path/class-list match SMIRK's src/utils/utils.py load_templates()
# exactly (quick_install.sh's expression_templates_famos.zip).
EXPRESSION_TEMPLATES_PATH = "assets/expression_templates_famos"
# SMIRK's own encoder config only ever used num_expression=50, so that's all their
# fitting pipeline saved per template frame - less than our FLAME_EXPRESSION_DIM=100.
# model/losses/cycle.py's template-injection augmentation zeroes dims beyond this.
EXPRESSION_TEMPLATE_NUM_DIMS = 50
EXPRESSION_TEMPLATE_CLASSES: tuple[str, ...] = (
    "lips_back",
    "rolling_lips",
    "mouth_side",
    "kissing",
    "high_smile",
    "mouth_up",
    "mouth_middle",
    "mouth_down",
    "blow_cheeks",
    "cheeks_in",
    "jaw",
    "lips_up",
)

# Cycle loss inner weights (model/losses/cycle.py): expression_cycle_loss bundles
# MSE(expression) + MSE(jaw)*10 + MSE(eyelid)*10, matching SMIRK's actual code (not
# just its paper's narrower Eq. 2, which covers expression only) - the plan's single
# outer "cycle 10" weight (Sec 6) is calibrated against this bundle as a whole.
CYCLE_EXPRESSION_WEIGHT = 1.0
CYCLE_JAW_WEIGHT = 10.0
CYCLE_EYELID_WEIGHT = 10.0
# Outer weight scaling expression_cycle_loss's whole bundle above (Sec 6 weights
# paragraph: "Starting loss weights (from SMIRK): cycle 10 ..."), applied in
# Stage 2 Pass B's training loop, distinct from the inner per-term weights.
CYCLE_LOSS_WEIGHT = 1.0
# Sec 6 weights paragraph: "For losses with no published anchor, starting
# guesses ...: beta identity cycle consistency 10 (mirroring the expression
# cycle weight, per SMIRK's 'similar to Eq. 2')" - restored to that literal
# spec value (was left at 1.0, same as CYCLE_LOSS_WEIGHT, since the plan's
# original release). Note this alone is what grows id_cycle's actual share of
# Pass B's total: raising CYCLE_LOSS_WEIGHT and IDENTITY_CYCLE_LOSS_WEIGHT
# together in lockstep (both to 10, matching spec literally) would leave their
# RATIO - and so id_cycle's share - unchanged, since id_cycle's raw magnitude
# is already ~40x smaller than expr_cycle's regardless of equal outer weights.
# Measured real Pass B logs (steps ~27000-54000 across all four stage2 AB
# lines): id_cycle was ~2% of Pass B's total at 1.0, ~18% at this 10.0 (with
# CYCLE_LOSS_WEIGHT left unchanged) - a real, present share without swamping
# expr_cycle.
IDENTITY_CYCLE_LOSS_WEIGHT = 10.0

# MICA shape distillation (model/mica/, model/losses/mica_shape.py, Sec 6d-9).
# Checkpoint path matches quick_install.sh's download location. Arcface's expected
# input size (a tightly aligned face crop, NOT the same crop as the main SViT
# input - see model/mica/mica.py's forward() docstring) and feature dim are fixed
# by the pretrained architecture, not tunable.
MICA_CHECKPOINT_PATH = "assets/mica.tar"
MICA_IMAGE_SIZE = 112
MICA_ARCFACE_FEATURE_DIM = 512
# Sec 6 weights paragraph: "For losses with no published anchor, starting guesses
# ...: MICA shape distillation 1.0".
MICA_SHAPE_LOSS_WEIGHT = 1.0

# Emotion loss (model/emotion/, model/losses/emotion.py, Sec 6d-10). Checkpoint
# path matches the actual downloaded filename (deca-epoch=01-val_loss_total/
# dataloader_idx_0=1.27607644.ckpt). Unlike MICA, this network takes the SAME
# 224x224 crop as the main SViT/renderer pipeline (no separate alignment step).
EMOTION_CHECKPOINT_PATH = "assets/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt"
EMOTION_IMAGE_SIZE = SVIT_IMG_SIZE
# Sec 6 weights paragraph: "Starting loss weights (from SMIRK): cycle 10,
# landmark 100, VGG 10, photometric 1, emotion 1." Doubled to 2.0 from that
# literal SMIRK value (unlike landmark/VGG, this one had no prior deviation) -
# a deliberate departure to push more perceptual/expression-accuracy signal
# into Pass A, alongside the reduced regularization pressure (REG_EXPRESSION_
# WEIGHT) - real Pass A logs put emotion's share of the combined total at
# ~5% at weight 1.0, ~10% at this 2.0.
EMOTION_LOSS_WEIGHT = 2.0

# Photometric (model/losses/photometric.py's photometric_loss, an L1) and VGG
# perceptual (VGGPerceptualLoss) losses, Stage 2 Pass A's reconstruction path
# only (no photometric supervision in Stage 1 - no rendering happens there).
# Sec 6 weights paragraph originally: "Starting loss weights (from SMIRK):
# ... VGG 10, photometric 1 ..." - SMIRK's own weight, calibrated against
# SMIRK's own co-active losses, not this project's. VGG_LOSS_WEIGHT lowered
# in two steps after measuring its actual share of Pass A/C's gradient: with
# VGG's raw magnitude (summed, not averaged, over 4 feature blocks - see
# VGGPerceptualLoss's own docstring) around 1.4-2.0 vs. landmark/mesh/mica/
# closure's raw magnitudes around 0.02-0.2, weight 10 gave VGG ~91% of Pass
# A's total loss (landmark/mesh/mica/closure/lvc combined were under 9%),
# alongside observed landmark/mesh accuracy regressing relative to the Stage
# 1 checkpoint Stage 2 started from. First dropped 10x to 1.0 (~50% share) -
# still the largest single term - then dropped again to 0.1 (10 -> 1 -> 0.1
# overall, 100x from the SMIRK-derived original), landing VGG at ~9% of Pass
# A's gradient and ~13% of Pass C's - closer to landmark/mica/mesh's
# individual shares than dominating them.
PHOTOMETRIC_LOSS_WEIGHT = 1.0
VGG_LOSS_WEIGHT = 0.1

# Temporal smoothness (model/losses/temporal_smoothness.py), Stage 2 Pass C
# only. velocity_penalty (L2, mean squared first difference) applies
# uniformly to expression/eyelid, jaw, camera+global-rotation, and shape
# params (see Sec 6's loss table). Originally 1.0, lowered 10x after
# observing Stage 2 training collapse to a near-fixed mesh with a prior
# acceleration+L2 formulation: quadratic growth on real motion, combined with
# this weight, over-suppressed genuine expressiveness (keep smoothness weight
# low and raise only if jitter persists; over-weighting damps mouthings).
# Switched from a velocity+L1 formulation back to L2 - since this is again a
# squared (not absolute) penalty, watch for the same over-suppression risk if
# jitter suppression looks too aggressive.
TEMPORAL_VELOCITY_WEIGHT = 0.1

# GT landmark precompute (dataset_processing/dataloading/landmark_cache.py, Sec
# 5.3). MediaPipe's own FaceLandmarker model asset - a data file (not code),
# copied from baselines/smirk_experiments/assets/face_landmarker.task, same as
# FLAME_MASK_AREA_WEIGHTS's source assets. Not the same file as
# FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH above, which is the curated 105-point
# index/barycentric-coordinate mapping, not a detector model.
MEDIAPIPE_TASK_MODEL_PATH = "assets/face_landmarker.task"
