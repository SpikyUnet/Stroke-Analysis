import torch
import torch.nn as nn
from pytorch_msssim import ssim

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3):
        """
        Hybrid Loss: Combines Mean Squared Error (MSE) and Structural Similarity Index Measure (SSIM).

        :param alpha: Weight for MSE loss
        :param beta: Weight for SSIM loss
        """
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss()

    def forward(self, pred, target):
        """
        Compute the weighted sum of MSE loss and (1 - SSIM loss).

        :param pred: Predicted tensor
        :param target: Ground truth tensor
        :return: Weighted hybrid loss
        """
        mse = self.mse_loss(pred, target)
        ssim_loss = 1 - ssim(pred, target, data_range=1.0)
        return self.alpha * mse + self.beta * ssim_loss
