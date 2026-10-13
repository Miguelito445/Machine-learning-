# Lab 02 Report

This report documents the work completed in Lab 02.

## Summary
This program was made in order to design implement and validate 
a two-hidden-layer neural network from scratch for handwritten digit classification.
It also work to Load load_digits dataset, normalize pixel inputs by dividing by 16, one-hot encoded target labels, and execute a 70/30 stratified split with random_state=42.
with this we observed the Training model using a mini-batch gradient descent (batch size 32, 50 epochs) across learning rates [0.001, 0.01, 0.1] and hidden widths [16, 64], saving the resulting loss curves to results/loss_curves.png

## Notes
Timing Analysis: Benchmarking revealed that the backward pass dominates execution time, which links directly to the higher volume of transposed matrix operations and gradient matrix multiplications required during backpropagation.
