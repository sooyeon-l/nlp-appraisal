import torch
import numpy as np

def evaluate_model(model, dataloader, target_dims, device): 
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad(): 
        for batch in dataloader:
            x = batch['input_ids'].to(device)
            y = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(x, attention_mask)
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())
        
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    per_dim_rmse = np.sqrt(((all_preds - all_labels) ** 2).mean(axis=0))
    per_dim_mae = np.abs(all_preds - all_labels).mean(axis=0)
    per_sample_loss = ((all_preds - all_labels) ** 2).mean(axis=1)

    return {
        'per_dim_rmse': dict(zip(target_dims, per_dim_rmse)),
        'per_dim_mae': dict(zip(target_dims, per_dim_mae)),
        'per_sample_loss': per_sample_loss,
        'all_preds': all_preds,
        'all_labels': all_labels,
    }