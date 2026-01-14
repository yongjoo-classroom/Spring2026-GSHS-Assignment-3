import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_input_tensors() -> tuple:
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y = np.array([[0], [1], [1], [0]], dtype=np.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x_tensor, y_tensor

def implement_xor() -> nn.Module:
    set_seed(42)
    X, Y = get_input_tensors()
    input_dim = 2
    hidden_dim = 8
    output_dim = 1
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
        nn.Sigmoid(),
    )
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.1)
    epochs = 3000
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()
    return model
