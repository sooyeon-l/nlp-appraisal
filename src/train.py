import os
import argparse
import json
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
from src.config import (
    SAVE_PATH, MODEL_NAME, TARGET_DIMS, BATCH_SIZE, N_EPOCHS, EARLY_STOPPING_PATIENCE, LR_LINEAR, SCHEDULER_FACTOR, SCHEDULER_PATIENCE
)


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, required=True, help='Run name e.g. phase1, phase2')
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(SAVE_PATH, f'best_model_{args.run}.pt')
    training_log_path = os.path.join(SAVE_PATH, f'training_log_{args.run}.csv')
    

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
    optimizer = AdamW([
        {'params': model.linear.parameters(), 'lr': LR_LINEAR}
    ])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',                             # minimize val_loss
        factor=SCHEDULER_FACTOR,                # halve the LR
        patience=SCHEDULER_PATIENCE,            # after 2 epochs without improvement
    )

    for epoch in range(N_EPOCHS): 
        all_preds = []
        all_labels = []
        model.train()
        for batch in tqdm(train_loader, leave=False): 
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            weights = batch['weights'].to(device)
            optimizer.zero_grad()
            outputs = model(x, attention_mask)
            loss = weighted_mse_loss(outputs, y, weights)
            loss.backward()
            optimizer.step()
            loss_record.append(loss.item())
        
        avg_train_loss = sum(loss_record[-len(train_loader):]) / len(train_loader)
        print(f"Epoch {epoch + 1}/{N_EPOCHS}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                y = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(x, attention_mask)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
        # Compute sample-level validation metrics from concatenated arrays
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        avg_val_loss = ((all_preds - all_labels) ** 2).mean()
        per_dim_rmse = np.sqrt(((all_preds - all_labels) ** 2).mean(axis=0))

        print(f"Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("New best model saved.")
            epochs_without_improvement = 0  
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

       
        # Use target_dims to print per_dim_rmse as a labeled table
        print("Per-dimension validation RMSE:")
        for dim, rmse in zip(TARGET_DIMS, per_dim_rmse):
            print(f"{dim}: {rmse:.4f}")
        # Save the epoch results to a CSV file including: epoch number, avg_train_loss, avg_val_loss, and per-dimension RMSE
        epoch_results = {
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
        }
        epoch_results.update({f'{dim}_rmse': rmse for dim, rmse in zip(TARGET_DIMS, per_dim_rmse)})
        results_df = pd.DataFrame([epoch_results])
        results_df.to_csv(training_log_path, mode='a', header=not os.path.exists(training_log_path), index=False)
    # Save batch loss record for plotting
    pd.Series(loss_record).to_csv(
        os.path.join(SAVE_PATH, f'batch_losses_{args.run}.csv'), 
        index=False, 
        header=['batch_loss']
    )

if __name__ == "__main__":
    main()
