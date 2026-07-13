"""Constants for the tokenized ViT encoder and FLAME parameterization (implementation-plan.md Sec 2)."""

# FLAME parameter dimensions
FLAME_SHAPE_DIM = 300
FLAME_EXPRESSION_DIM = 100
NUM_EYELID_PARAMS = 2
FLAME_JAW_POSE_DIM = 3

# Camera + global rotation (Sec 2.2): 1 scale + 3 global rotation + 3 translation
CAMERA_SCALE_DIM = 1
GLOBAL_ROTATION_DIM = 3
TRANSLATION_DIM = 3

# Component token parameter-group dims (output of each per-token MLP head, Sec 2.2/2.3)
SHAPE_TOKEN_DIM = FLAME_SHAPE_DIM
EXPRESSION_TOKEN_DIM = FLAME_EXPRESSION_DIM + NUM_EYELID_PARAMS
JAW_TOKEN_DIM = FLAME_JAW_POSE_DIM
CAMERA_TOKEN_DIM = CAMERA_SCALE_DIM + GLOBAL_ROTATION_DIM + TRANSLATION_DIM

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

# Attention bias window (Sec 4.2)
TT_WINDOW_SIZE = 11
