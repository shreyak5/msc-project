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

# Attention bias window (Sec 4.2)
TT_WINDOW_SIZE = 11

# Attention bias (Sec 4.4): bias(i,j,h) = m_h * s_tilde_j - n_h * |i-j|. Heads span a
# genuine grid of (m, n) combinations (not a single m=k*n line): 4 m-values x 2
# n-values = 8 heads (TT_NUM_HEADS). Both are geometric series:
# - m: spans up to ~50, so that an extreme score deviation (s_tilde ~ 0.1) can
#   dominate over a typical distance penalty even at the grid's low end.
# - n: ALiBi's own first two slopes from its standard 8-head geometric sequence
#   (2^-1, 2^-2) - see Press et al., "Train Short, Test Long".
TT_BIAS_M_VALUES: tuple[float, ...] = (6.25, 12.5, 25.0, 50.0)
TT_BIAS_N_VALUES: tuple[float, ...] = (0.5, 0.25)

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
# ~1% randomly retained face pixels"). Matches SMIRK's actual trainer config
# (config_train.yaml: mask_ratio: 0.01, mask_dilation_radius: 10). FLAME_masks_triangles
# is a curated map of FLAME-region-name -> triangle indices, used to bias which
# triangles are eligible to be sampled as retained pixels (Sec 2.5's ablation note:
# keep the 1% ratio - 5% breaks expression control).
MASK_RATIO = 0.01
MASK_DILATION_RADIUS = 10
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
# separate per-parameter-group constants (so each can be tuned independently later),
# but all currently equal - TokenFace uses a single uniform weight across all FLAME
# params (shape/expression/jaw), unlike SMIRK's own differentiated per-parameter
# weights; matches TokenFace here since it's the primary architecture reference.
REG_EXPRESSION_WEIGHT = 1e-4
REG_JAW_WEIGHT = 1e-4
REG_SHAPE_WEIGHT = 1e-4

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
# only shape the region-weighted average *within* the mesh loss itself).
MESH_LOSS_LAMBDA = 2.0
VERTEX_CONSISTENCY_LOSS_LAMBDA = 1.2

# Landmark loss (model/losses/landmark.py, Sec 6). Sec 6 weights paragraph:
# "Starting loss weights (from SMIRK): ... landmark 100 ...". SMIRK's own code
# applies this weight to both the FAN and MediaPipe terms independently, then
# sums them (not averaged) - matched here. Deliberately lowered from SMIRK's
# literal 100 to 10 for this project: compared only against the losses actually
# co-active during Stage 1 pretraining (mesh 2.0, Lvc 1.2, MICA 1.0 - VGG/cycle/
# photometric/emotion aren't active until Stage 2), literal 100 would be a
# 50-100x outlier rather than just "somewhat higher" - 10 keeps landmark's
# intended priority (geometry correctness leads Stage 1, per its own stated
# purpose) without that scale of disparity. Revisit once real loss curves are
# available (Sec 6: "All weights are config knobs").
LANDMARK_LOSS_WEIGHT = 10.0
# Eye/lip closure terms have no SMIRK precedent at all (an EMOCA-derived
# addition, Sec 6d-3) - reusing the same weight to start rather than introducing
# a second untuned constant.
CLOSURE_LOSS_WEIGHT = LANDMARK_LOSS_WEIGHT

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
# landmark 100, VGG 10, photometric 1, emotion 1."
EMOTION_LOSS_WEIGHT = 1.0

# GT landmark precompute (dataset_processing/dataloading/landmark_cache.py, Sec
# 5.3). MediaPipe's own FaceLandmarker model asset - a data file (not code),
# copied from baselines/smirk_experiments/assets/face_landmarker.task, same as
# FLAME_MASK_AREA_WEIGHTS's source assets. Not the same file as
# FLAME_MEDIAPIPE_LMK_EMBEDDING_PATH above, which is the curated 105-point
# index/barycentric-coordinate mapping, not a detector model.
MEDIAPIPE_TASK_MODEL_PATH = "assets/face_landmarker.task"
