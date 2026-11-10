# Lab 05 Report

This report documents the work completed in Lab 05.

## Summary
Summary1. Value Node Design: Implemented a core Value class that stores data values, gradient accumulations, parent nodes, and local backward derivative functions to build a dynamic computation graph. 
2. Graph Construction & Operations: Implemented fundamental operators (addition, multiplication, powers, and ReLU) that dynamically construct computation graphs during the forward pass.   
3. Backpropagation Engine: Implemented the backward pass mechanism by topologically sorting the graph, initializing the root gradient to 1, traversing nodes in reverse order, and correctly accumulating gradients rather than overwriting them.   
4. Gradient Verification: Verified custom autograd engine gradients against PyTorch autograd and finite-difference numerical estimates on a composite function, ensuring the maximum absolute difference remains below $1e-6$.   
5. Tiny Model Training: Trained a tiny multi-layer network ($2\to8\to1$ with a sigmoid activation and Binary Cross-Entropy loss) on the make_moons dataset, benchmarking execution speed against native PyTorch over multiple runs.   
6. Execution Modes & Tensor Contiguity: Inspected graph structures in PyTorch, compared eager execution performance against compiled or traced execution (such as torch.compile or torch.jit), and analyzed how non-contiguous tensor layouts impact operation execution speeds.

## Notes
Custom Autograd Dynamics: Building an autograd engine from scratch explicitly demonstrates how computation graphs record operations dynamically, mapping forward computations directly to reverse-mode differentiation rules.
Gradient Accumulation: Proper gradient accumulation (+=) is critical during backpropagation because nodes with multiple downstream paths (fan-out) receive gradient contributions from each path, which must be summed up correctly.
Compiled vs. Eager Execution: Eager execution incurs Python interpreter and dispatch overheads for every individual operation. Compiling or tracing graphs (torch.compile / torch.jit) fuses operations and optimizes kernel execution, substantially reducing overhead for repetitive architectures.

Tensor Contiguity Impact: Non-contiguous tensors (created via slicing or transposing) disrupt memory stride patterns, causing cache misses and forcing memory layout copies or inefficient stride-walking during computational kernels, which increases execution latency.
