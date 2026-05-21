"""
Quick CPU-friendly training run.
Reduces epochs, batch size, and num_workers to finish in ~10-15 min on CPU.
Imports everything from model.py.
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys

# Import from the main model module
from model import (
    DenoisingDataset, ImprovedUNet, PriorityFocusedLoss,
    EarlyStopping, evaluate_model, save_metrics,
    plot_training_curves, calculate_model_size
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np


def train_quick(num_epochs=3, batch_size=2):
    """Quick training run optimized for CPU."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    output_dir = 'denoising_results'
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Smaller images for speed
        transforms.ToTensor(),
    ])

    try:
        train_dataset = DenoisingDataset('Denoising_Dataset_prepared', transform, 'Train')
        val_dataset = DenoisingDataset('Denoising_Dataset_prepared', transform, 'Val')

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Model
        model = ImprovedUNet(skip_connections=[False, True, True, True]).to(device)
        model_info = calculate_model_size(model)
        print(f"Model parameters: {model_info['parameters']:,}")
        print(f"Model size: {model_info['size_mb']:.1f} MB")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}")
        print("-" * 50)

        # Training setup
        criterion = PriorityFocusedLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2, eta_min=1e-7)

        best_metrics = {'defect_ssim': 0, 'epoch': 0}
        history = {'train_loss': [], 'val_metrics': []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0

            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
            for batch in pbar:
                degraded, clean, mask = [x.to(device) for x in batch]

                if mask.shape[1] == 3:
                    mask = mask.mean(dim=1, keepdim=True)

                optimizer.zero_grad()
                outputs = model(degraded, mask)
                loss = criterion(outputs, clean, mask)

                if torch.isnan(loss):
                    print(f"NaN loss at epoch {epoch}")
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = train_loss / len(train_loader)

            # Validation
            model.eval()
            val_metrics = evaluate_model(model, val_loader, device)

            scheduler.step(epoch)

            # Save best
            defect_ssim = val_metrics.get('defect_ssim', 0)
            if np.isnan(defect_ssim):
                defect_ssim = 0
            if defect_ssim > best_metrics['defect_ssim']:
                best_metrics = {
                    'defect_ssim': defect_ssim,
                    'epoch': epoch,
                    'state_dict': model.state_dict()
                }
                torch.save(best_metrics, 'best_model.pth')

            history['train_loss'].append(avg_train_loss)
            history['val_metrics'].append(val_metrics)

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Metrics:")
            for name, val in val_metrics.items():
                print(f"    {name}: {val:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save results
        save_metrics(history, best_metrics['epoch'], output_dir)
        plot_training_curves(history)

        print(f"\nTraining complete!")
        print(f"Best epoch: {best_metrics['epoch']}")
        print(f"Best defect SSIM: {best_metrics['defect_ssim']:.4f}")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    train_quick(num_epochs=epochs, batch_size=batch)
