import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import DETECTORS


@DETECTORS.register_module()
class CustomEventPointHead(nn.Module):
    def __init__(self, input_channels=100, output_points=500):  # Match input_channels to the actual input
        super(CustomEventPointHead, self).__init__()
        
        input_channels=100
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(128 * 32 * 32, output_points * 2)  # Correct input size for Linear layer
        self.output_points = output_points

    def forward(self, x):
        """Predict (x, y) coordinates from input feature maps."""
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        x = x.view(-1, self.output_points, 2)  # Reshape to (batch_size, num_points, 2)

        # Convert tensor to a list of tuples for each sample in the batch
        pred_coords = [list(map(tuple, sample.cpu().tolist())) for sample in x]
        return pred_coords


    def forward_train(self, x, gt_event_points):
        """
        Compute predictions and losses for event points during training.

        Args:
            x: Input feature maps, shape (batch_size, C, H, W).
            gt_event_points: List of ground truth lists of (x, y) coordinates.

        Returns:
            dict[str, Tensor]: A dictionary containing the loss for event points.
        """
        # Forward pass
        pred_coords = self.forward(x)
        batch_size = len(gt_event_points)
        total_loss = 0.0

        for pred, gt in zip(pred_coords, gt_event_points):
            # Convert ground truth to tensor
            gt_tensor = torch.tensor(gt, device=x.device, dtype=torch.float32)
            mask = torch.ones(len(gt), dtype=torch.float32, device=x.device)

            # Handle padding for ground truth
            if len(gt) < self.output_points:
                mask = F.pad(mask, (0, self.output_points - len(gt)), value=0)  # Extend mask with zeros
                gt_tensor = F.pad(gt_tensor, (0, 0, 0, self.output_points - len(gt)), value=0)  # Pad with (0, 0)
            elif len(gt) > self.output_points:
                mask = mask[:self.output_points]  # Truncate mask
                gt_tensor = gt_tensor[:self.output_points]  # Truncate ground truth

            # Convert predictions to tensor
            pred_tensor = torch.tensor(pred, device=x.device, dtype=torch.float32)

            # Compute loss with mask
            loss = F.mse_loss(pred_tensor, gt_tensor, reduction='none')  # Element-wise loss
            loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()  # Apply mask and normalize by valid points
            total_loss += loss

        avg_loss = total_loss / batch_size
        return {'loss_event_points': avg_loss}


