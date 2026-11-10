import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torchinfo import summary
from thop import profile
import matplotlib.pyplot as plt


# 1. Dataset Preparation (Fashion-MNIST)
def get_dataloaders(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    full_train_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Split official train 60k into 55k train and 5k validation
    train_dataset, val_dataset = random_split(
        full_train_dataset, [55000, 5000],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader

# 2. Model Definitions
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 7 * 7, 10)
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()
        # Split 28x28 image into 16 patches of 7x7. Each patch has dimension 7*7 = 49.
        self.patch_size = 7
        self.d_model = 64
        self.linear_proj = nn.Linear(49, self.d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(self.d_model, 10)
        
    def forward(self, x):
        B, C, H, W = x.shape  # B, 1, 28, 28
        # Extract 16 patches of 7x7
        # Shape: [B, 1, 4 patches H, 7, 4 patches W, 7] -> [B, 16, 49]
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(B, 16, C * self.patch_size * self.patch_size)
        
        x = self.linear_proj(x)  # [B, 16, 64]
        x = self.transformer_encoder(x)  # [B, 16, 64]
        x = x.mean(dim=1)  # Mean pooling across patches -> [B, 64]
        x = self.fc(x)  # [B, 10]
        return x


# 3. Cost Calculation & Benchmarking Latency
def measure_latency(model, device, batch_size=1, num_runs=100):
    model.eval()
    dummy_input = torch.randn(batch_size, 1, 28, 28).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in xrange(10) if 'xrange' in globals() else range(10):
            _ = model(dummy_input)
            
    latencies = []
    with torch.no_grad():
        for _ in range(num_runs):
            start_time = time.time()
            _ = model(dummy_input)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000.0)  # ms
            
    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    return median_latency

# 4. Training Loop
def train_model(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
        
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

# Main Execution Pipeline
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=128)
    
    models = {
        "MLP": MLPModel(),
        "CNN": CNNModel(),
        "Attention": AttentionModel()
    }
    
    results = {}
    
    dummy_input = torch.randn(1, 1, 28, 28).to(device)
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        model.to(device)
        
        # Parameter count via torchinfo
        stat = summary(model, input_size=(1, 1, 28, 28), verbose=0)
        param_count = stat.total_params
        
        # FLOPs via thop
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # MACs to FLOPs conversion
        
        # Latency measurements
        lat_b1 = measure_latency(model, device, batch_size=1, num_runs=100)
        lat_b128 = measure_latency(model, device, batch_size=128, num_runs=20)
        
        print(f"Parameters: {param_count}")
        print(f"FLOPs: {flops}")
        print(f"Latency (Batch size 1): {lat_b1:.2f} ms")
        print(f"Latency (Batch size 128): {lat_b128:.2f} ms")
        
        # Train and test
        print(f"Training {name}...")
        train_model(model, train_loader, val_loader, device, epochs=5)
        test_acc = evaluate_model(model, test_loader, device)
        print(f"Test Accuracy: {test_acc:.2f}%")
        
        results[name] = {
            "params": param_count,
            "flops": flops,
            "lat_b1": lat_b1,
            "lat_b128": lat_b128,
            "accuracy": test_acc
        }

    # 5. Plotting Trade-offs
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]
    params = [results[n]["params"] for n in names]
    latencies = [results[n]["lat_b128"] for n in names]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(params, accuracies, color=['blue', 'green', 'red'], s=100)
    for i, txt in enumerate(names):
        plt.annotate(txt, (params[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Parameter Count")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Parameters")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(latencies, accuracies, color=['blue', 'green', 'red'], s=100)
    for i, txt in enumerate(names):
        plt.annotate(txt, (latencies[i], accuracies[i]), fontsize=12, xytext=(5, 5), textcoords='offset points')
    plt.xlabel("Inference Latency Batch 128 (ms)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Accuracy vs Latency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("model_tradeoffs.png")
    print("\nSaved trade-off plots to model_tradeoffs.png successfully!")