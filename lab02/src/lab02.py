#task 1
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

digits = load_digits()
X = digits.data / 16.0
y_int = digits.target

num_classes = 10
Y = np.zeros((y_int.size, num_classes))
Y[np.arange(y_int.size), y_int] = 1

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.3, stratify=y_int, random_state=42
)
#task 2

def initialize_parameters(input_dim, hidden_dim, output_dim):
    np.random.seed(42)
    parameters = {
        "W1": np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim),
        "b1": np.zeros((hidden_dim,)),
        "W2": np.random.randn(hidden_dim, hidden_dim) * np.sqrt(2.0 / hidden_dim),
        "b2": np.zeros((hidden_dim,)),
        "W3": np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim),
        "b3": np.zeros((output_dim,))
    }
    return parameters

#task 3
def relu(Z):
    return np.maximum(0, Z)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def forward_propagation(X, params):
    Z1 = np.dot(X, params["W1"]) + params["b1"]
    A1 = relu(Z1)

    Z2 = np.dot(A1, params["W2"]) + params["b2"]
    A2 = relu(Z2)
    
    Z3 = np.dot(A2, params["W3"]) + params["b3"]
    A3 = softmax(Z3)
    
    cache = (X, Z1, A1, Z2, A2, Z3, A3)
    return A3, cache

#task 4

def compute_loss(Y_hat, Y):
    m = Y.shape[0]
    Y_hat_clipped = np.clip(Y_hat, 1e-15, 1 - 1e-15)
    loss = -np.sum(Y * np.log(Y_hat_clipped)) / m
    return loss

#task 5

def backward_propagation(params, cache, Y):
    X, Z1, A1, Z2, A2, Z3, A3 = cache
    m = Y.shape[0]
    
    dZ3 = A3 - Y
    dW3 = np.dot(A2.T, dZ3) / m
    db3 = np.sum(dZ3, axis=0) / m
    
    dA2 = np.dot(dZ3, params["W3"].T)
    dZ2 = dA2 * (Z2 > 0)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0) / m

    dA1 = np.dot(dZ2, params["W2"].T)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0) / m
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2, "dW3": dW3, "db3": db3}
    return grads

def gradient_check(params, X_sample, Y_sample, grads, epsilon=1e-5):
    pass_count = 0
    for key in ["W1", "W2", "W3"]:
        flat_indices = np.random.choice(params[key].size, 3, replace=False)
        for idx in flat_indices:
            orig_val = params[key].flat[idx]
            
            params[key].flat[idx] = orig_val + epsilon
            A_plus, _ = forward_propagation(X_sample, params)
            loss_plus = compute_loss(A_plus, Y_sample)
            
            params[key].flat[idx] = orig_val - epsilon
            A_minus, _ = forward_propagation(X_sample, params)
            loss_minus = compute_loss(A_minus, Y_sample)
            
            params[key].flat[idx] = orig_val
            
            num_grad = (loss_plus - loss_minus) / (2 * epsilon)
            an_grad = grads["d" + key].flat[idx]
            
            rel_error = np.abs(num_grad - an_grad) / np.max([np.abs(num_grad), np.abs(an_grad), 1e-12])
            if rel_error < 1e-6:
                pass_count += 1
    print(f"Gradient check passed for {pass_count} tested weights (Required: at least 9 of 10).")


    import time

def benchmark_passes():
    params = initialize_parameters(64, 64, 10)
    X_sample = X_train[:32]
    Y_sample = y_train[:32]
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(100):
  
        start = time.perf_counter()
        A3, cache = forward_propagation(X_sample, params)
        fwd_times.append(time.perf_counter() - start)
        
  
        start = time.perf_counter()
        _ = backward_propagation(params, cache, Y_sample)
        bwd_times.append(time.perf_counter() - start)
        
    print(f"Median Forward Pass Time: {np.median(fwd_times)*1000:.4f} ms[cite: 1]")
    print(f"Median Backward Pass Time: {np.median(bwd_times)*1000:.4f} ms[cite: 1]")

benchmark_passes()

results_summary = {}
learning_rates = [0.001, 0.01, 0.1]
hidden_widths = [16, 64]
batch_size = 32
epochs = 50

for W in hidden_widths:
    for lr in learning_rates:
        params = initialize_parameters(64, W, 10)
        loss_history = []
        
        for epoch in range(epochs):
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            Y_shuffled = y_train[permutation]
            
            epoch_loss = 0.0
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))
            
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                Y_batch = Y_shuffled[start_idx:end_idx]
                
                A3, cache = forward_propagation(X_batch, params)
                loss = compute_loss(A3, Y_batch)
                epoch_loss += loss * X_batch.shape[0]
                
                grads = backward_propagation(params, cache, Y_batch)
                
                for key in params:
                    params[key] -= lr * grads["d" + key]
            
            epoch_loss /= X_train.shape[0]
            loss_history.append(epoch_loss)
        
        A_test, _ = forward_propagation(X_test, params)
        test_preds = np.argmax(A_test, axis=1)
        test_labels = np.argmax(y_test, axis=1)
        test_acc = np.mean(test_preds == test_labels)
        
        results_summary[(W, lr)] = {"loss": loss_history, "accuracy": test_acc}

import os
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))
for (W, lr), data in results_summary.items():
    plt.plot(data["loss"], label=f"W={W}, lr={lr} (Acc: {data['accuracy']:.2f})")
plt.xlabel("Epochs")
plt.ylabel("Cross-Entropy Loss")
plt.title("Loss Curves across Learning Rates and Hidden Widths")
plt.legend()
plt.grid(True)
plt.savefig("results/loss_curves.png")
plt.close()

print("Training completed across all configurations. Loss curves saved to results/loss_curves.png.")