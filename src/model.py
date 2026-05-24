import torch
import torch.nn as nn

# Model class takes the base model, single linear layer, and applies sigmoid activation to get output in [0,1] range for each of the 15 dimensions to match normalized targets
class AppraisalModel(nn.Module): 
    def __init__(self, base_model, num_labels=15):
        super(AppraisalModel, self).__init__()
        self.base_model = base_model
        self.linear = nn.Linear(self.base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_output = outputs.last_hidden_state[:, 0, :]
        # Switched to average pooling
        pooled_output = outputs.last_hidden_state.mean(dim=1)
        sigmoid_output = torch.sigmoid(self.linear(pooled_output))

        return sigmoid_output