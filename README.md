# Autonomous Drone Neural Engine (NumPy From Scratch)

## Overview
This repository contains a custom Neural Network layer built entirely from scratch using **NumPy** and pure mathematics. It bypasses pre-built libraries like Scikit-Learn or TensorFlow to demonstrate a fundamental understanding of the applied mathematics powering Machine Learning algorithms.

The engine ingests raw drone telemetry (Wind Speed, Rotor RPM, Altitude) and predicts flight stability by mathematically adjusting its own internal weights using Gradient Descent.

## Core Mathematical Pillars Demonstrated
1. **Tensors & Matrix Slicing:** Representing multi-dimensional drone telemetry as 2D arrays and extracting specific feature vectors using matrix coordinates.
2. **Array Broadcasting & Variance:** Standardizing the dataset (subtracting the mean, dividing by standard deviation) across the entire matrix instantly without `for` loops.
3. **Linear Algebra (Dot Products):** Executing the forward pass of a neural network by multiplying the input matrix by the weight vector (`np.dot`).
4. **Calculus (Gradients & Partial Derivatives):** Calculating the Mean Squared Error (MSE) and applying partial derivatives to find the gradient. The engine then uses this gradient to update the weights, actively reducing the error rate.

## Technology Stack
* **Language:** Python 3.x
* **Computation Engine:** NumPy
