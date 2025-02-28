import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import Preprocessor
from dataset import CustomDataset
from model import spikyUNet
from loss import HybridLoss
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio, MeanSquaredError
import numpy as np

# Single flag for selecting input format
use_preprocessed_x = False  # If True, use CLAHE & Morphology; If False, use original X

# Load data
input_folder = "/path/to/dicom/inputs"
output_folder = "/path/to/dicom/outputs"
preprocessor = Preprocessor(input_folder, output_folder, preprocess=use_preprocessed_x)
X_data, Y_data = preprocessor.run()

# Split into train/val/test
from sklearn.model_selection import train_test_split
X_train, X_temp, Y_train, Y_temp = train_test_split(X_data, Y_data, test_size=0.2, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)

#  Save the test set for later use in `test.py`
np.save("X_test.npy", X_test)
np.save("Y_test.npy", Y_test)
print("Saved X_test.npy and Y_test.npy")

# Create datasets & dataloaders
train_dataset = CustomDataset(X_train, Y_train)
val_dataset = CustomDataset(X_val, Y_val)

train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=5, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = spikyUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize loss function and metrics
criterion = HybridLoss(alpha=0.7, beta=0.3)  
ssim_metric = StructuralSimilarityIndexMeasure().to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
mse_metric = MeanSquaredError().to(device)

best_val_loss = float("inf")

# Training loop with validation
for epoch in range(100):
    model.train()
    train_loss = 0.0
    train_ssim, train_psnr, train_mse = 0.0, 0.0, 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Compute metrics
        train_loss += loss.item()
        train_ssim += ssim_metric(outputs, targets).item()
        train_psnr += psnr_metric(outputs, targets).item()
        train_mse += mse_metric(outputs, targets).item()

    # Compute average metrics
    train_loss /= len(train_loader)
    train_ssim /= len(train_loader)
    train_psnr /= len(train_loader)
    train_mse /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    val_ssim, val_psnr, val_mse = 0.0, 0.0, 0.0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            val_loss += loss.item()
            val_ssim += ssim_metric(outputs, targets).item()
            val_psnr += psnr_metric(outputs, targets).item()
            val_mse += mse_metric(outputs, targets).item()

    # Compute average validation metrics
    val_loss /= len(val_loader)
    val_ssim /= len(val_loader)
    val_psnr /= len(val_loader)
    val_mse /= len(val_loader)

    print(f"Epoch {epoch+1}:")
    print(f"  Train Loss: {train_loss:.4f} | SSIM: {train_ssim:.4f} | PSNR: {train_psnr:.4f} | MSE: {train_mse:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | SSIM: {val_ssim:.4f} | PSNR: {val_psnr:.4f} | MSE: {val_mse:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        print(f"Saving best model with val loss {val_loss:.4f}")
        torch.save(model.state_dict(), "best_model.pth")
        best_val_loss = val_loss
