import numpy as np


def retailer_profit(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.int)
    y_pred = np.array(y_pred, dtype=np.int)
    cost = sum((y_true & y_pred) * 5  # Detected fraud
               + (y_true & ~y_pred) * -5  # Undetected fraud
               + (~y_true & y_pred) * -25)  # Wrong detect as fraud
    return cost
