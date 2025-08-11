import torch
import torch.nn as nn
import logging

class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def forward(self, pred, target, prev_value=None):
        # 차원 맞추기
        if len(target.shape) == 1:
            target = target.view(-1, 1)
        if len(pred.shape) == 1:
            pred = pred.view(-1, 1)
        
        # MSE Loss
        mse_loss = self.mse(pred, target)
        
        # Directional Loss (차원 확인)
        if pred.shape[1] > 1:
            pred_diff = pred[:, 1:] - pred[:, :-1]
            target_diff = target[:, 1:] - target[:, :-1]
            directional_loss = -torch.mean(torch.sign(pred_diff) * torch.sign(target_diff))
        else:
            directional_loss = torch.tensor(0.0).to(pred.device)
        
        # Continuity Loss
        continuity_loss = 0
        if prev_value is not None:
            if len(prev_value.shape) == 1:
                prev_value = prev_value.view(-1, 1)
            continuity_loss = self.mse(pred[:, 0:1], prev_value)
        
        return self.alpha * mse_loss + (1 - self.alpha) * directional_loss + self.beta * continuity_loss