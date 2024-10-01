import torch
from torch.nn import CrossEntropyLoss
from trl import SFTTrainer


import torch
from torch.nn import CrossEntropyLoss

class WeightedCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, reduction='mean'):
        super().__init__(weight=weight, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction  # Add this line to store the reduction method

    def forward(self, input, target, per_token_weights=None):
        # Compute standard CrossEntropyLoss without reduction
        loss = super(WeightedCrossEntropyLoss, self).forward(input, target)  # Shape: [batch_size * seq_len]
        
        # Apply per-token weights if provided
        if per_token_weights is not None:
            loss = loss * per_token_weights.view(-1)
        
        # Reduce the loss to a scalar
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # Be cautious with 'none' reduction

        
class WeightedLossSFTTrainer(SFTTrainer):
    def __init__(self, *args, weight=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.loss_fct = WeightedCrossEntropyLoss(reduction='mean')  # Ensure reduction is 'mean'

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tensors
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Create per-token weights
        batch_size, seq_len = labels.shape
        per_token_weights = torch.ones_like(shift_labels, dtype=shift_logits.dtype)

        # Apply higher weight to the label token (second-to-last token)
        seq_length = seq_len - 1  # Adjusted for shifted labels
        label_token_positions = torch.arange(batch_size, device=labels.device) * seq_length + (seq_length - 2)
        per_token_weights[label_token_positions] = self.weight

        # Compute the loss using the custom loss function
        loss = self.loss_fct(shift_logits, shift_labels, per_token_weights)

        # Ensure loss is scalar (should be if reduction='mean')
        if not torch.is_tensor(loss) or loss.dim() != 0:
            raise ValueError(f"Loss should be a scalar, but got loss of shape {loss.shape}")

        if return_outputs:
            return loss, outputs
        else:
            return loss


