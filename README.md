# MLP for XOR

In this assignment, you will implement a Multi-Layer Perceptron (MLP) to learn the XOR function by training a neural network that can model non-linear decision boundaries.

## Assignment Instructions

### Objective
Implement the `implement_xor` and `get_input_tensors` function in `mlp.py` to train a neural network that correctly classifies the XOR dataset.

1. The `get_input_tensors` function should:
   - Fill in the missing output values for the XOR truth table based on the given input pairs.
   - Convert the input and output lists into tensors (refer to the provided sample code).
   - Return the resulting tensors as tuple `(x_tensor, y_tensor)`.

2. The `implement_xor` function should:
   - Most of the starter code is already provided. Complete the sections marked as: ### Implement your code here
   - Fill the input, hidden, and output layer dimensions to complete the MLP model.
   - Experiment with different dimensions for the hidden layer (e.g., 2, 4, 8) to observe their effect on learning the XOR function.
   - In the training loop, feed the input tensors through the model to obtain predictions.

## Testing
Given the trained MLP model, pass the inputs from the XOR truth table and verify that the predicted outputs match the expected XOR labels.