import numpy as np


def retailer_profit(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cost = sum((y_true & y_pred) * 5    # Detected fraud
           + (y_true & ~y_pred) * -5    # Undetected fraud
           + (~y_true & y_pred) * -25)  # Wrong detect as fraud
    return cost
