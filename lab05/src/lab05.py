import os
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

print("Starting Lab 05 execution...")

# Step 1 & 2: Value Node Design & Graph Construction Operators
class Value:
    """A scalar value node that tracks computation graphs for automatic differentiation."""
    def __init__(self, data, children=(), op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(children)
        self.op = op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for simplicity"
        out = Value(self.data ** other, (self,), f'**{other}')
        
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')
        
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return Value(other) + (-1 * self)

    def __truediv__(self, other):
        return self * (other ** -1)

# Step 3: Backward Pass Engine (Topological Sort & Accumulation)
    def backward(self):
        """Performs topological sort and propagates gradients backward without overwriting."""
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# Step 4: Verification Against PyTorch and Numerical Gradients
def verify_gradients():
    print("--- Step 4: Verifying Custom Autograd vs PyTorch ---")
    a_val = Value(2.0)
    b_val = Value(-3.0)
    c_val = Value(10.0)
    
    out_custom = ((a_val * b_val + c_val) ** 2).relu()
    out_custom.backward()
    
    a_pt = torch.tensor(2.0, requires_grad=True)
    b_pt = torch.tensor(-3.0, requires_grad=True)
    c_pt = torch.tensor(10.0, requires_grad=True)
    
    out_pt = ((a_pt * b_pt + c_pt) ** 2).relu()
    out_pt.backward()
    
    diff_a = abs(a_val.grad - a_pt.grad.item())
    diff_b = abs(b_val.grad - b_pt.grad.item())
    diff_c = abs(c_val.grad - c_pt.grad.item())
    
    print(f"Custom grad a: {a_val.grad}, PyTorch grad a: {a_pt.grad.item()}")
    print(f"Custom grad b: {b_val.grad}, PyTorch grad b: {b_pt.grad.item()}")
    print(f"Custom grad c: {c_val.grad}, PyTorch grad c: {c_pt.grad.item()}")
    assert max(diff_a, diff_b, diff_c) < 1e-6, "Gradient verification failed! Difference exceeds 1e-6"
    print("Gradient verification passed successfully (max abs difference < 1e-6).\n")

# Step 5: Tiny Model Training on make_moons
class TinyPyTorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x))))

def train_tiny_model():
    print("--- Step 5: Training Tiny Model on make_moons ---")
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_t = torch.FloatTensor(X_train)
    y_t = torch.FloatTensor(y_train).unsqueeze(1)
    
    model = TinyPyTorchModel()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    runs = 5
    latencies = []
    
    for run in range(runs):
        start_time = time.time()
        for epoch in range(50):
            optimizer.zero_grad()
            preds = model(X_t)
            loss = criterion(preds, y_t)
            loss.backward()
            optimizer.step()
        elapsed = (time.time() - start_time) * 1000.0
        latencies.append(elapsed)
        
    latencies.sort()
    median_time = latencies[len(latencies) // 2]
    print(f"PyTorch Tiny Model Training Complete. Median time over {runs} runs: {median_time:.2f} ms.\n")

# Step 6: PyTorch Execution Modes & Tensor Contiguity
def benchmark_execution_and_contiguity():
    print("--- Step 6: PyTorch Execution Modes & Tensor Contiguity ---")
    
    class CompilableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(100, 100),
                nn.ReLU(),
                nn.Linear(100, 10)
            )
        def forward(self, x):
            return self.net(x)

    model = CompilableModel().eval()
    dummy_input = torch.randn(64, 100)
    
    # Warm-up eager execution
    for _ in range(10):
        _ = model(dummy_input)
        
    # Eager timing
    start = time.time()
    for _ in range(1000):
        _ = model(dummy_input)
    eager_time = (time.time() - start) * 1000.0
    print(f"Eager Execution Time: {eager_time:.2f} ms")
    
    # Traced / Compiled timing
    compiled_time = 0.0
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        for _ in range(10):
            _ = traced_model(dummy_input)
            
        start = time.time()
        for _ in range(1000):
            _ = traced_model(dummy_input)
        compiled_time = (time.time() - start) * 1000.0
        print(f"Traced/Compiled Execution Time: {compiled_time:.2f} ms")
    except Exception as e:
        print(f"Compilation/Tracing skipped or failed: {e}")

    # Tensor Contiguity test
    contiguous_tensor = torch.randn(1000, 1000)
    non_contiguous_tensor = contiguous_tensor.t()
    
    print(f"Is contiguous tensor contiguous? {contiguous_tensor.is_contiguous()}")
    print(f"Is non-contiguous tensor contiguous? {non_contiguous_tensor.is_contiguous()}")
    
    start = time.time()
    for _ in range(5000):
        _ = torch.sum(contiguous_tensor)
    cont_time = (time.time() - start) * 1000.0
    
    start = time.time()
    for _ in range(5000):
        _ = torch.sum(non_contiguous_tensor)
    non_cont_time = (time.time() - start) * 1000.0
    
    print(f"Contiguous summation time: {cont_time:.2f} ms")
    print(f"Non-contiguous summation time: {non_cont_time:.2f} ms")

    # Plot Generation using an absolute path so it saves directly in your folder
    try:
        import matplotlib.pyplot as plt
        modes = ['Eager Execution', 'Traced / Compiled']
        times = [eager_time, compiled_time]

        plt.figure(figsize=(6, 4))
        plt.bar(modes, times, color=['skyblue', 'salmon'])
        plt.ylabel('Execution Time (ms for 1000 runs)')
        plt.title('Performance Comparison: Eager vs Compiled Execution')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        output_plot_path = r"C:\Users\erict\Desktop\lab05\execution_timings_comparison.png"
        plt.savefig(output_plot_path)
        plt.close()
        print(f"Saved execution timing plot successfully to: {output_plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot generation.")

if __name__ == "__main__":
    verify_gradients()
    train_tiny_model()
    benchmark_execution_and_contiguity()
    print("\nLab 05 script completed successfully!")