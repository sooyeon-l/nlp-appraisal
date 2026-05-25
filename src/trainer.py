import os
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from src.loss import weighted_mse_loss

def train_model(model, optimizer, scheduler, train_loader, val_loader, 
                n_epochs, patience, checkpoint_path, training_log_path, 
                target_dims, device, grad_clip=1.0):
    
    loss_record = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(n_epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            loss_record.append(loss.item())
        
        avg_train_loss = sum(loss_record[-len(train_loader):]) / len(train_loader)
        print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {avg_train_loss:.4f}")


        model.eval()
        with torch.no_grad(): 
            for batch in val_loader:
                x = batch['input_ids'].to(device)
                y = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                outputs = model(x, attention_mask)
                all_preds.append(outputs.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        avg_val_loss = ((all_preds - all_labels) ** 2).mean()
        per_dim_rmse = np.sqrt(((all_preds - all_labels) ** 2).mean(axis=0))

        print(f"Validation Loss: {avg_val_loss:.4f}")
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch, 
                'model_state_dict': model.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(), 
                'best_val_loss': float(best_val_loss), 
            }, checkpoint_path)
            print("New best model saved.")
            epochs_without_improvement = 0
        else: 
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience: 
                print("Early stopping triggered.")
                break
        
        print("Per-dimension validation RMSE:")
        for dim, rmse in zip(target_dims, per_dim_rmse): 
            print(f"{dim}: {rmse:.4f}")

        epoch_results = {
            'epoch': epoch + 1, 
            'avg_train_loss': avg_train_loss, 
            'avg_val_loss': avg_val_loss, 
        }
        epoch_results.update({f'{dim}_rmse': rmse for dim, rmse in zip(target_dims, per_dim_rmse)})
        results_df = pd.DataFrame([epoch_results])
        results_df.to_csv(training_log_path, mode='a', header=not os.path.exists(training_log_path), index=False)
    
    return loss_record