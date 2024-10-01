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
    def __init__(self, *args, weight=2.0, label_marker_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        if label_marker_tokens is None:
            self.label_marker_tokens = [19890, 3801, 1190, 781] # token IDs for the label marker "Prediction #\n"
        else:
            self.label_marker_tokens = label_marker_tokens
        self.loss_fct = WeightedCrossEntropyLoss(reduction='mean')

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift logits and labels for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten the tensors
        batch_size, seq_length = shift_labels.shape
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)

        # Create per-token weights
        per_token_weights = torch.ones_like(shift_labels, dtype=shift_logits.dtype)

        # Create a mask for padding tokens
        non_padding_mask = shift_labels != self.loss_fct.ignore_index
        per_token_weights = per_token_weights * non_padding_mask.float()

        if self.label_marker_tokens is not None:
            # Reshape shift_labels back to (batch_size, seq_length)
            shift_labels_reshaped = shift_labels.view(batch_size, seq_length)

            # Identify positions of the label marker in each sequence
            label_token_positions = []
            for i in range(batch_size):
                sequence = shift_labels_reshaped[i]
                # Convert to list for easier indexing
                sequence_list = sequence.tolist()
                # Find the position where the marker appears
                for idx in range(len(sequence_list) - len(self.label_marker_tokens)):
                    if sequence_list[idx:idx + len(self.label_marker_tokens)] == self.label_marker_tokens:
                        # The label token is immediately after the marker
                        label_pos = idx + len(self.label_marker_tokens)
                        label_token_positions.append(i * seq_length + label_pos)
                        break
            # Apply higher weight to the label tokens
            if label_token_positions:
                per_token_weights[label_token_positions] = self.weight
        else:
            # If label_marker_tokens is not provided, default to the last non-padding token
            shift_labels_reshaped = shift_labels.view(batch_size, seq_length)
            label_token_positions = []
            for i in range(batch_size):
                sequence = shift_labels_reshaped[i]
                # Find the position of the last non-padding token
                non_padding_indices = (sequence != self.loss_fct.ignore_index).nonzero(as_tuple=True)[0]
                if len(non_padding_indices) > 0:
                    label_pos = non_padding_indices[-1].item()
                    label_token_positions.append(i * seq_length + label_pos)
            # Apply higher weight to the label tokens
            if label_token_positions:
                per_token_weights[label_token_positions] = self.weight

        # Compute the loss using the custom loss function
        loss = self.loss_fct(shift_logits, shift_labels, per_token_weights)

        if return_outputs:
            return loss, outputs
        else:
            return loss


