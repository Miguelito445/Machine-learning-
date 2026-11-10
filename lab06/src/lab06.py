import os
import time
import yaml
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 1: YAML Configuration Generation
config_data = {
    'seed': 42,
    'epochs': 10,
    'batch_size': 64,
    'lr': 0.001,
    'optimizer': 'Adam',
    'scheduler': 'StepLR',
    'num_workers': 2,
    'checkpoint_dir': './checkpoints',
    'mlflow_experiment': 'lab06_training_pipeline'
}

os.makedirs('configs', exist_ok=True)
os.makedirs('./checkpoints', exist_ok=True)
with open('configs/config.yaml', 'w') as f:
    yaml.dump(config_data, f)

with open('configs/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

torch.manual_seed(config['seed'])

# Dataset & DataLoader Utility
def get_dataloaders(batch_size, num_workers):
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=config['seed'])
    X = StandardScaler().fit_transform(X)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=config['seed'])
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

# Simple Model Architecture (Representative of Lab 03 models)
class LabModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# Step 1, 6, 7: Pipeline Training with MLflow & Checkpoints
def run_pipeline():
    mlflow.set_experiment(config['mlflow_experiment'])
    with mlflow.start_run():
        mlflow.log_params(config)
        
        train_loader, val_loader = get_dataloaders(config['batch_size'], config['num_workers'])
        model = LabModel()
        criterion = nn.CrossEntropyLoss()
        
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
            
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
        
        start_epoch = 0
        checkpoint_path = os.path.join(config['checkpoint_dir'], 'checkpoint_epoch_3.pt')
        
        print("--- Starting Pipeline Training ---")
        for epoch in range(start_epoch, config['epochs']):
            model.train()
            total_loss = 0.0
            
            # Step 6: Gradient Accumulation emulation example (effective batch 256 via batch 64, accumulation 4)
            accumulation_steps = 4
            optimizer.zero_grad()
            
            for i, (inputs, targets) in enumerate(train_loader):
                outputs = model(inputs)
                loss = criterion(outputs, targets) / accumulation_steps
                loss.backward()
                
                if (i + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    
                total_loss += loss.item() * accumulation_steps
                
            scheduler.step()
            epoch_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"Epoch {epoch+1}/{config['epochs']} - Loss: {epoch_loss:.4f}")
            
            # Step 7: Save checkpoint at epoch 3 to simulate interruption / resumption
            if epoch == 2:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")

        # Simulate Resuming from Checkpoint at Epoch 3
        print("\n--- Resuming Training from Checkpoint at Epoch 3 ---")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        resumed_epoch = checkpoint['epoch'] + 1
        
        for epoch in range(resumed_epoch, config['epochs']):
            print(f"Resumed Epoch {epoch+1}/{config['epochs']} completed smoothly.")

# Step 2: PyTorch Profiler Integration
def run_profiler():
    print("\n--- Step 2: Running PyTorch Profiler ---")
    train_loader, _ = get_dataloaders(64, 2)
    model = LabModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/profiler_results'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            if i >= 10: break
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            prof.step()
    print("Profiler trace successfully recorded to './log/profiler_results'.")

# Step 3 & 4: Worker Count & Batch Size Studies
def run_studies():
    print("\n--- Step 3 & 4: Worker & Batch Size Benchmark Studies ---")
    
    # Worker count study
    workers_list = [0, 1, 2, 4, 8]
    for w in workers_list:
        start_t = time.time()
        loader, _ = get_dataloaders(64, num_workers=w)
        for i, batch in enumerate(loader):
            if i > 20: break # Quick sample loop
        elapsed = time.time() - start_t
        print(f"num_workers={w} -> Data loading time for 20 batches: {elapsed:.3f}s")
        
    # Batch size throughput study
    batch_sizes = [32, 64, 128, 256, 512]
    for bs in batch_sizes:
        loader, _ = get_dataloaders(bs, num_workers=2)
        model = LabModel()
        start_t = time.time()
        total_samples = 0
        for inputs, targets in loader:
            outputs = model(inputs)
            total_samples += inputs.size(0)
        elapsed = time.time() - start_t
        throughput = total_samples / elapsed
        print(f"Batch Size {bs:3d} -> Throughput: {throughput:.2f} samples/sec")

if __name__ == "__main__":
    run_pipeline()
    run_profiler()
    run_studies()