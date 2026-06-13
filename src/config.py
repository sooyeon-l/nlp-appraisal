from pathlib import Path

# Paths

SAVE_PATH = Path("/workspace/data")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "roberta-base"
TEXT_COLUMN = "generated_text"
MAX_LENGTH = 128


# crowd-enVENT target dimensions

TARGET_DIMS = [
    "suddenness",
    "familiarity",
    "predict_event",
    "pleasantness",
    "unpleasantness",
    "goal_relevance",
    "chance_responsblt",
    "self_responsblt",
    "other_responsblt",
    "predict_conseq",
    "goal_support",
    "urgency",
    "self_control",
    "other_control",
    "chance_control",
    "accept_conseq",
    "standards",
    "social_norms",
    "attention",
    "not_consider",
    "effort",
]


# Scherer CPM objective grouping

OBJECTIVE_GROUPS = {
    "relevance": [
        "suddenness",
        "familiarity",
        "predict_event",
        "attention",
        "pleasantness",
        "unpleasantness",
        "goal_relevance",
        "not_consider",
    ],
    "implication": [
        "chance_responsblt",
        "self_responsblt",
        "other_responsblt",
        "predict_conseq",
        "goal_support",
        "urgency",
    ],
    "coping": [
        "self_control",
        "other_control",
        "chance_control",
        "accept_conseq",
        "effort",
    ],
    "normative": [
        "standards",
        "social_norms",
    ],
}


# Architecture options

MODEL_TYPES = [
    "flat_linear",
    "flat_mlp",
    "grouped_parallel",
    "grouped_sequential",
]


# Loss options

LOSS_TYPES = [
    "mse",
    "weighted_mse",
    "group_balanced_mse",
    "weighted_group_balanced_mse",
]


# Model hyperparameters

SHARED_DIM = 256
GROUP_HIDDEN_DIM = 128
DROPOUT_P = 0.2


# Training hyperparameters

BATCH_SIZE = 16
N_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5

LR_HEAD = 1e-3
LR_BASE_MODEL = 2e-5
LR_HEAD_FINETUNE = 1e-4

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2

GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42


# Safety checks

grouped_dims = [
    dim
    for group_dims in OBJECTIVE_GROUPS.values()
    for dim in group_dims
]

if len(grouped_dims) != len(set(grouped_dims)):
    raise ValueError("A target dimension appears in more than one CPM group.")

if set(grouped_dims) != set(TARGET_DIMS):
    missing = set(TARGET_DIMS) - set(grouped_dims)
    unexpected = set(grouped_dims) - set(TARGET_DIMS)

    raise ValueError(
        "OBJECTIVE_GROUPS does not exactly cover TARGET_DIMS.\n"
        f"Missing: {sorted(missing)}\n"
        f"Unexpected: {sorted(unexpected)}"
    )