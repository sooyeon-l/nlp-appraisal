import os
import argparse
import json
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModel
from src.model import AppraisalModel, freeze_roberta_layers, confirm_trainable_status
from src.dataset import AppraisalDataset
from src.loss import weighted_mse_loss
from src.trainer import train_model
from src.config import (
    DROPOUT_P, SAVE_PATH, MODEL_NAME, TARGET_DIMS, BATCH_SIZE, N_EPOCHS, EARLY_STOPPING_PATIENCE, LR_LINEAR, SCHEDULER_FACTOR, SCHEDULER_PATIENCE, RANDOM_SEED, GRAD_CLIP, WEIGHT_DECAY
)


def main(): 
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, help='Run name e.g. phase1, phase2')
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(SAVE_PATH, f'best_model_{args.run}.pt')
    training_log_path = os.path.join(SAVE_PATH, f'training_log_{args.run}.csv')
    
    config_snapshot = {
        'model_name': MODEL_NAME, 'batch_size': BATCH_SIZE,
        'n_epochs': N_EPOCHS, 'lr_base': 'NONE',
        'lr_linear': LR_LINEAR, 'seed': RANDOM_SEED, 'dropout': 'NONE', 
        'weight_decay': WEIGHT_DECAY
    }
    pd.DataFrame([config_snapshot]).to_csv(
        os.path.join(SAVE_PATH, f'config_{args.run}.csv'), index=False
    )

    loss_record = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    weights_json = json.load(open(os.path.join(SAVE_PATH, 'dim_weights.json'), 'r'))
    roberta = AutoModel.from_pretrained(MODEL_NAME)
    model = AppraisalModel(roberta, num_labels=len(TARGET_DIMS))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = AppraisalDataset(os.path.join(SAVE_PATH, 'train.csv'), tokenizer, weights_json, TARGET_DIMS)
    val_dataset = AppraisalDataset(os.path.join(SAVE_PATH, 'val.csv'), tokenizer, weights_json, TARGET_DIMS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)  

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    freeze_roberta_layers(model)
    confirm_trainable_status(model)
    optimizer = AdamW(
        model.linear.parameters(), 
        lr=LR_LINEAR, 
        weight_decay=WEIGHT_DECAY
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',                             # minimize val_loss
        factor=SCHEDULER_FACTOR,                # halve the LR
        patience=SCHEDULER_PATIENCE,            # after 2 epochs without improvement
    )

    loss_record = train_model(model, optimizer, scheduler, train_loader, val_loader, N_EPOCHS, EARLY_STOPPING_PATIENCE, checkpoint_path, training_log_path, TARGET_DIMS, device, grad_clip=1.0)

    pd.Series(loss_record).to_csv(
        os.path.join(SAVE_PATH, f'batch_losses_{args.run}.csv'), 
        index=False, 
        header=['batch_loss']
    )

if __name__ == "__main__":
    main()
