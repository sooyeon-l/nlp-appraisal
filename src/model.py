import torch
import torch.nn as nn

# Model class takes the base model, single linear layer, and applies sigmoid activation to get output in [0,1] range for each of the 15 dimensions to match normalized targets
class AppraisalModel(nn.Module): 
    def __init__(self, base_model, num_labels=15, dropout_p=0.1):
        super(AppraisalModel, self).__init__()
        self.base_model = base_model
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, 0, :]
        # Switched to average pooling
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        pooled_output = self.dropout(pooled_output)
        sigmoid_output = torch.sigmoid(self.linear(pooled_output))

        return sigmoid_output

def freeze_roberta_layers(model): 
    for param in model.base_model.parameters():
        param.requires_grad = False

def unfreeze_roberta_layers(model):
    for param in model.base_model.parameters():
        param.requires_grad = True

def confirm_trainable_status(model):
   
    base_model_params = [p.requires_grad for p in model.base_model.parameters()]
    base_model_trainable = sum(base_model_params)
    base_model_total = len(base_model_params)
    print(f"Base model trainable parameters: {base_model_trainable}/{base_model_total} ({base_model_trainable / base_model_total * 100:.2f}%)")

    linear_params = [p.requires_grad for p in model.linear.parameters()]
    linear_trainable = sum(linear_params)
    linear_total = len(linear_params)
    print(f"Linear layer trainable parameters: {linear_trainable}/{linear_total} ({linear_trainable / linear_total * 100:.2f}%)")