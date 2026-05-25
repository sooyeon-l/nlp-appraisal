import os

# Path
SAVE_PATH = '/workspace/data' 

# Model
MODEL_NAME = 'roberta-base'

# Data
TARGET_DIMS = [
    'goal_relevance', 'self_responsblt', 'other_responsblt', 'chance_responsblt',
    'goal_support', 'predict_conseq', 'urgency', 'self_control', 'other_control',
    'chance_control', 'accept_conseq', 'social_norms', 'standards', 'attention', 'effort'
]

# Training
BATCH_SIZE = 16
N_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
LR_BASE_MODEL = 2e-5
LR_LINEAR = 1e-3
LR_BASE_MODEL_P2 = 2e-5
LR_LINEAR_P2 = 1e-4
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
DROPOUT_P = 0.1