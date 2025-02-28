import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from model import spikyUNet
from loss import HybridLoss
import numpy as np
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MeanSquaredError

# Load test data
X_test = np.load("X_data_preprocessed.npy")
Y_test = np.load("Y_data_preprocessed.npy")
test_dataset = CustomDataset(X_test, Y_test)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = spikyUNet().to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Use the same loss function and metrics
criterion = HybridLoss(alpha=0.7, beta=0.3)
ssim_metric = StructuralSimilarityIndexMeasure().to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
mse_metric = MeanSquaredError().to(device)

# Testing
test_loss = 0.0
test_ssim, test_psnr, test_mse = 0.0, 0.0, 0.0

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        test_loss += loss.item()
        test_ssim += ssim_metric(outputs, targets).item()
        test_psnr += psnr_metric(outputs, targets).item()
        test_mse += mse_metric(outputs, targets).item()

# Compute average test metrics
test_loss /= len(test_loader)
test_ssim /= len(test_loader)
test_psnr /= len(test_loader)
test_mse /= len(test_loader)

print("\n********** Test Results **********")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test SSIM: {test_ssim:.4f}")
print(f"Test PSNR: {test_psnr:.4f}")
print(f"Test MSE: {test_mse:.4f}")
