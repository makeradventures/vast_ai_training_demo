import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from pathlib import Path
import wandb
import os
import sys
import signal
import time
from datetime import datetime

class TrainingConfig:
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 3e-4
        self.epochs = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "vit_tiny_patch16_224"
        # Use workspace directory for vast.ai
        self.checkpoint_dir = "/workspace/vast_ai_training_demo/checkpoints"
        self.log_interval = 50
        
def setup_wandb(config):
    try:
        wandb.init(
            project="cifar10-vit-vast",
            config={
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "model": config.model_name,
                "epochs": config.epochs,
                "device": str(config.device)
            }
        )
    except Exception as e:
        print(f"Warning: Could not initialize W&B: {e}")
        return False
    return True

def get_dataloaders(batch_size):
    try:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        
        transform = transforms.Compose([
            transforms.Resize(224),  # ViT requires 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        
        # Set download directory explicitly for vast.ai
        data_dir = "/workspace/vast_ai_training_demo/data"
        os.makedirs(data_dir, exist_ok=True)
        
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, 
                                       download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, 
                                      download=True, transform=transform)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                                shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                               shuffle=False, num_workers=4, pin_memory=True)
        
        return train_loader, test_loader
    except Exception as e:
        print(f"Error setting up data loaders: {e}")
        sys.exit(1)

def load_checkpoint(model, optimizer, checkpoint_dir):
    try:
        checkpoint_path = Path(checkpoint_dir) / "latest.pt"
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
            return start_epoch
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
    return 0

def save_checkpoint(model, optimizer, epoch, checkpoint_dir, is_final=False):
    try:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_path = Path(checkpoint_dir) / "latest.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, checkpoint_path)
        
        if is_final:
            final_path = Path(checkpoint_dir) / f"final_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, final_path)
            print(f"Saved final checkpoint to {final_path}")
    except Exception as e:
        print(f"Warning: Could not save checkpoint: {e}")

def train_epoch(model, train_loader, criterion, optimizer, config, epoch):
    model.train()
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.device), target.to(config.device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % config.log_interval == 0:
            # Calculate speed
            elapsed = time.time() - start_time
            images_per_sec = (batch_idx + 1) * len(data) / elapsed
            
            # Log progress
            progress = {
                "train_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx,
                "images_per_second": images_per_sec
            }
            
            try:
                wandb.log(progress)
            except Exception:
                pass
            
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]'
                  f'\tLoss: {loss.item():.6f}'
                  f'\tSpeed: {images_per_sec:.1f} images/sec')

def evaluate(model, test_loader, criterion, config, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    metrics = {
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "epoch": epoch
    }
    
    try:
        wandb.log(metrics)
    except Exception:
        pass
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')
    
    return accuracy

def setup_signal_handling(model, optimizer, config):
    def signal_handler(signum, frame):
        print("\nSignal received. Saving checkpoint and exiting...")
        save_checkpoint(model, optimizer, epoch, config.checkpoint_dir, is_final=True)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

def main():
    # Print start time and system info
    print(f"Starting training at {datetime.now()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    config = TrainingConfig()
    
    # Setup W&B logging
    setup_wandb(config)
    
    try:
        # Create model
        model = timm.create_model(config.model_name, pretrained=True, num_classes=10)
        model = model.to(config.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Get data
        train_loader, test_loader = get_dataloaders(config.batch_size)
        
        # Load checkpoint if exists
        start_epoch = load_checkpoint(model, optimizer, config.checkpoint_dir)
        
        # Setup signal handling for graceful shutdown
        setup_signal_handling(model, optimizer, config)
        
        best_accuracy = 0
        
        # Training loop
        for epoch in range(start_epoch, config.epochs):
            print(f"\nStarting epoch {epoch} at {datetime.now()}")
            
            train_epoch(model, train_loader, criterion, optimizer, config, epoch)
            accuracy = evaluate(model, test_loader, criterion, config, epoch)
            
            # Save checkpoint
            save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                try:
                    best_path = Path(config.checkpoint_dir) / "best.pt"
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'accuracy': accuracy
                    }, best_path)
                    print(f"Saved new best model with accuracy {accuracy:.2f}%")
                except Exception as e:
                    print(f"Warning: Could not save best model: {e}")
        
        # Save final checkpoint
        save_checkpoint(model, optimizer, config.epochs-1, config.checkpoint_dir, is_final=True)
        print(f"Training completed at {datetime.now()}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        sys.exit(1)
    
    try:
        wandb.finish()
    except Exception:
        pass

if __name__ == "__main__":
    main()
