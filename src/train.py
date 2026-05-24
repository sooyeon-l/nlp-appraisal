import os
import json
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AppraisalModel
from dataset import AppraisalDataset    
from loss import weighted_mse_loss
from transformers import AutoTokenizer
from transformers import AutoModel
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

SAVE_PATH = '/content/drive/MyDrive/2026-1/NLP/appraisal/nlp-appraisal/data' # TODO: make this easy to change because it is environment-specific 
checkpoint_path = os.path.join(SAVE_PATH, 'best_model.pt')
training_log_path = os.path.join(SAVE_PATH, 'training_log.csv')

target_dims = [
    'goal_relevance', 'self_responsblt', 'other_responsblt', 'chance_responsblt',
    'goal_support', 'predict_conseq', 'urgency', 'self_control', 'other_control',
    'chance_control', 'accept_conseq', 'social_norms', 'standards', 'attention', 'effort'
]

model_name = 'roberta-base'
batch_size = 16
early_stopping_patience = 3
learning_rates = {
    'base_model': 2e-5,
    'linear_layer': 1e-3
}
scheduler_factor = 0.5
scheduler_patience = 2
n_epoch = 10


def main(): 

    loss_record = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0

    weights_json = json.load(open(os.path.join(SAVE_PATH, 'dim_weights.json'), 'r'))
    roberta = AutoModel.from_pretrained(model_name)
    model = AppraisalModel(roberta, num_labels=len(target_dims))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = AppraisalDataset(os.path.join(SAVE_PATH, 'train.csv'), tokenizer, weights_json, target_dims)
    val_dataset = AppraisalDataset(os.path.join(SAVE_PATH, 'val.csv'), tokenizer, weights_json, target_dims)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model.to(device)
    optimizer = AdamW([
        {'params': model.base_model.parameters(), 'lr':learning_rates['base_model']},
        {'params': model.linear.parameters(), 'lr': learning_rates['linear_layer']}
    ])

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',                             # minimize val_loss
        factor=scheduler_factor,                # halve the LR
        patience=scheduler_patience,            # after 2 epochs without improvement
    )

    for epoch in range(n_epoch): 
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
        print(f"Epoch {epoch + 1}/{n_epoch}, Training Loss: {avg_train_loss:.4f}")

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                y = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(x, attention_mask)
                # Unweighted MSE for validation to get a clearer signal of true performance
                val_loss = ((outputs - y) ** 2).mean()
                val_losses.append(val_loss.item())
                # Collect predictions and labels
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print("New best model saved.")
            epochs_without_improvement = 0  
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print("Early stopping triggered.")
                break

        # Per epoch, 1) overall training loss with weighted MSE, 2) overall validation loss with unweighted MSE, 3) per-dimension validation RMSE for all 15 dimensions are recorded in a table and saved as a CSV file for later analysis. 4) The best model checkpoint is saved based on validation loss.
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        per_dim_rmse = np.sqrt(((all_preds - all_labels) ** 2).mean(axis=0))
        # Use target_dims to print per_dim_rmse as a labeled table
        print("Per-dimension validation RMSE:")
        for dim, rmse in zip(target_dims, per_dim_rmse):
            print(f"{dim}: {rmse:.4f}")
        # Save the epoch results to a CSV file including: epoch number, avg_train_loss, avg_val_loss, and per-dimension RMSE
        epoch_results = {
            'epoch': epoch + 1,
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
        }
        epoch_results.update({f'{dim}_rmse': rmse for dim, rmse in zip(target_dims, per_dim_rmse)})
        results_df = pd.DataFrame([epoch_results])
        results_df.to_csv(os.path.join(SAVE_PATH, 'training_log.csv'), mode='a', header=not os.path.exists(os.path.join(SAVE_PATH, 'training_log.csv')), index=False)
if __name__ == "__main__":
    main()
