# Lab 03 Report

This report documents the work completed in Lab 03.

## Summary
## 1. 
Dataset: Loaded the Fashion-MNIST dataset in PyTorch with tensor conversion and normalization, split the official training set into training and validation subsets, and configured data loaders with a batch size of 128.
## 2. 
Model Implementation: Designed three distinct neural network architectures (MLP, CNN, and an Attention-based Transformer model) to target a comparable parameter size of 100k to 150k parameters.
## 3.
Cost Computation: Evaluated model costs by counting parameters via profiling tools, calculating theoretical layer FLOPs, comparing them against profiler metrics, and measuring median inference latency for batch sizes 1 and 128.
## 4.
Model Training: Trained all three model architectures for 5 epochs using the Adam optimizer with a learning rate of 1e-3 and recorded their final test set accuracies.
## 5.
Evaluation and Trade-off Analysis: Generated performance trade-off plots comparing accuracy against parameter counts and latency, utilizing architectural access patterns to evaluate cost-effectiveness.

## Notes
## Accuracy and Performance: 
The Convolutional Neural Network (CNN) typically achieves the best balance of high test accuracy, parameter efficiency, and low latency on image datasets like Fashion-MNIST due to its built-in spatial inductive biases. 
## Architectural Limitations of MLPs: 
While the Multi-Layer Perceptron (MLP) is straightforward to implement, flattening 2D image inputs destroys spatial hierarchies, resulting in lower parameter efficiency and weaker generalization compared to spatial models. 
## Attention Overhead: 
The Attention-based model captures global contextual relationships effectively, but its patch-based projection and self-service query-key-value mechanisms introduce higher computational overhead and increased inference latency, making it less cost-effective for small-scale image tasks.
## Memory Access Patterns: 
Convolutions benefit from localized, cache-friendly memory access patterns. In contrast, fully connected layers and transformer attention mechanisms rely heavily on global matrix multiplications and sequence-wide operations, putting higher pressure on memory bandwidth and increasing latency at smaller batch sizes.
