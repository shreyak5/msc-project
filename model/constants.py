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
