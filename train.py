# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm
from pathlib import Path
import wandb
import os
import argparse

class TrainingConfig:
    def __init__(self):
        self.batch_size = 128
        self.learning_rate = 3e-4
        self.epochs = 30
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "vit_tiny_patch16_224"
        self.checkpoint_dir = "/workspace/checkpoints"
        self.log_interval = 100
        
def setup_wandb(config):
    wandb.init(
        project="cifar10-vit",
        config={
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "model": config.model_name,
            "epochs": config.epochs
        }
    )

def get_dataloaders(batch_size):
    # CIFAR-10 mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    transform = transforms.Compose([
        transforms.Resize(224),  # ViT requires 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                   download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, 
                                  download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, test_loader

def load_checkpoint(model, optimizer, checkpoint_dir):
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")
        return start_epoch
    return 0

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(checkpoint_dir) / "latest.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

def train_epoch(model, train_loader, criterion, optimizer, config, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(config.device), target.to(config.device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % config.log_interval == 0:
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch,
                "batch": batch_idx
            })
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

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
    
    wandb.log({
        "test_loss": test_loss,
        "test_accuracy": accuracy,
        "epoch": epoch
    })
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} '
          f'({accuracy:.2f}%)\n')

def main():
    config = TrainingConfig()
    
    # Setup W&B logging
    setup_wandb(config)
    
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
    
    # Training loop
    for epoch in range(start_epoch, config.epochs):
        train_epoch(model, train_loader, criterion, optimizer, config, epoch)
        evaluate(model, test_loader, criterion, config, epoch)
        save_checkpoint(model, optimizer, epoch, config.checkpoint_dir)
    
    wandb.finish()

if __name__ == "__main__":
    main() 
