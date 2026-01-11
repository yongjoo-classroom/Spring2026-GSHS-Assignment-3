import torch
import numpy as np
from mlp import implement_xor

def test_xor_predictions():
    '''
    Tests the XOR predictions of the model.
    '''
    # inputs
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y_expected = np.array([[0], [1], [1], [0]])

    # get the predictions from the model
    model = implement_xor()
    with torch.no_grad():
        preds = model(torch.tensor(x, dtype=torch.float32))
    y_pred = [1 if p[0] >= 0.5 else 0 for p in preds]

    # Check 1: predictions are in the range [0, 1]
    for p in preds:
        assert 0.0 <= p[0] <= 1.0, f"Prediction out of range: {p[0]}"

    # Check 2: predictions match expected values
    assert list(y_pred) == list(y_expected.flatten()), \
            f"Predictions do not match expected values: {y_pred} != {y_expected.flatten()}"

