import torch
from torch.nn import CrossEntropyLoss
from trl import SFTTrainer

class WeightedLossSFTTrainer(SFTTrainer):
    def __init__(self, *args, weight=1.0, label_marker_tokens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight  # Weight for the classification loss
        if label_marker_tokens is None:
            self.label_marker_tokens = [19890, 3801, 1190, 781]  # Tokens for "Prediction #\n"
        else:
            self.label_marker_tokens = label_marker_tokens
        self.loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        self.classification_loss_fct = CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        # Remove labels from inputs to prevent the model from computing the loss internally
        labels = inputs.pop("labels")
        attention_mask = inputs.get("attention_mask")
        
        # Forward pass without labels
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Shift logits and labels for language modeling loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute language modeling loss
        lm_loss = self.loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        
        # Initialize classification loss
        classification_loss = torch.tensor(0.0).to(lm_loss.device)
        
        # Find label token positions and compute classification loss over full vocabulary
        batch_size, seq_length = labels.shape
        prediction_logits = []
        ground_truth_token_ids = []
        
        labels_cpu = labels.cpu().numpy()  # Move to CPU for processing
        
        for i in range(batch_size):
            sequence = labels_cpu[i]  # Labels for sequence i
            marker_length = len(self.label_marker_tokens)
            
            # Find the position of the label marker
            found_marker = False
            for idx in range(len(sequence) - marker_length):
                if list(sequence[idx:idx + marker_length]) == self.label_marker_tokens:
                    label_pos = idx + marker_length  # Position of the label token
                    found_marker = True
                    break
            
            if found_marker:
                # Get ground truth token ID at label position
                ground_truth_token_id = sequence[label_pos]
                ground_truth_token_ids.append(ground_truth_token_id)
                
                # Extract logits at label position over the full vocabulary
                label_logits = logits[i, label_pos, :]  # Shape: (vocab_size)
                prediction_logits.append(label_logits.unsqueeze(0))
            else:
                continue  # Handle cases where the marker isn't found
        
        if prediction_logits:
            # Stack prediction logits and ground truth labels
            prediction_logits = torch.cat(prediction_logits, dim=0)  # Shape: (batch_size, vocab_size)
            ground_truth_token_ids = torch.tensor(ground_truth_token_ids, dtype=torch.long).to(logits.device)
            
            # Compute classification loss over the full vocabulary
            classification_loss = self.classification_loss_fct(prediction_logits, ground_truth_token_ids)
            
            # Combine losses
            total_loss = lm_loss + self.weight * classification_loss
        else:
            total_loss = lm_loss

        # Detach losses before logging
        lm_loss_value = lm_loss.detach().item()
        classification_loss_value = classification_loss.detach().item()
        total_loss_value = total_loss.detach().item()

        # Log individual losses
        self.log({
            "lm_loss": lm_loss_value,
            "classification_loss": classification_loss_value,
            "total_loss": total_loss_value
        })

        if return_outputs:
            return total_loss, outputs
        else:
            return total_loss
            
    # def training_step(self, model, inputs):
    #     model.train()
    #     inputs = self._prepare_inputs(inputs)

    #     # Compute loss and get individual losses
    #     total_loss, losses_dict = self.compute_loss(model, inputs, return_outputs=True)

    #     # Log individual losses
    #     logs = {
    #         "train_lm_loss": losses_dict["lm_loss"].detach().item(),
    #         "train_classification_loss": losses_dict["classification_loss"].detach().item(),
    #         "train_total_loss": total_loss.detach().item()  # Detach total_loss before calling .item()
    #     }
    #     self.log(logs)

    #     return total_loss  # Return total_loss for backpropagation


    
    
    # def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=None):
    #     model.eval()
    #     with torch.no_grad():
    #         inputs = self._prepare_inputs(inputs)
    #         total_loss, losses_dict = self.compute_loss(model, inputs, return_outputs=True)

    #     # Log individual losses
    #     logs = {
    #         "eval_lm_loss": losses_dict["lm_loss"].detach().item(),
    #         "eval_classification_loss": losses_dict["classification_loss"].detach().item(),
    #         "eval_total_loss": total_loss.detach().item()  # Detach total_loss before calling .item()
    #     }
    #     self.log(logs)

    #     if prediction_loss_only:
    #         return (total_loss, None, None)

    #     return (total_loss, None, None)



