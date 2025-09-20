import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import VGG19_Weights

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features):
        device = features[0].device
        batch_size = features[0].shape[0]
        
        total_loss = 0
        for feat in features:
            feat = F.normalize(feat.view(batch_size, -1), dim=1)
            sim_matrix = torch.matmul(feat, feat.t()) / self.temperature
            pos_mask = torch.eye(batch_size, device=device)
            loss = -torch.log(
                torch.exp(sim_matrix * pos_mask).sum(1) /
                torch.exp(sim_matrix).sum(1)
            ).mean()
            
            total_loss += loss
            
        return total_loss / len(features)

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_1
            vgg[4:9],  # relu2_1
            vgg[9:18], # relu3_1
            vgg[18:27],# relu4_1
            vgg[27:36] # relu5_1
        ])
        for bl in self.blocks:
            for p in bl.parameters():
                p.requires_grad = False
                
    def forward(self, x, y):
        x_feats = []
        y_feats = []
        for block in self.blocks:
            x = block(x)
            y = block(y)
            x_feats.append(x)
            y_feats.append(y)
            
        return sum(F.mse_loss(xf, yf) for xf, yf in zip(x_feats, y_feats))
