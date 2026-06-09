# Base Task

Classifying Fashion-MNIST images using a custom PyTorch neural network.

## Architecture

The model follows the diagram provided in the task sheet exactly.

Input (28x28) -> Flatten (784)
-> Hidden Layer (784 -> 16)
-> Branch A: (16 -> 8 -> 8) with skip connection
-> Branch B: (16 -> 12 -> 8)
-> Concatenate (8 + 8 = 16)
-> Output (16 -> 10 classes)

## Key concepts used

Forward propagation, backward propagation, skip connections, ReLU activation,
dropout, Adam optimizer, learning rate scheduling, cross entropy loss,
train/val split, confusion matrix.

## Results

Validation accuracy: around 87-89% depending on run.
Plots are saved inside the notebook and as separate PNG files.

## Files

notebooks/spider_ml_task1_base.ipynb - full training pipeline
saved_models/fashion_net_weights.pkl - model weights saved with Pickle
saved_models/fashion_net_state_dict.pth - PyTorch state dict backup
submission.csv - test set predictions