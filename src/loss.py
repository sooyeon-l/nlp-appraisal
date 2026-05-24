# Explicitly defining this function with precomputed weights from only training data to avoid data leakage
def weighted_mse_loss(preds, labels, weights):
    squared_error = (preds - labels) ** 2
    weighted_squared_error = squared_error * weights
    loss = weighted_squared_error.mean()
    return loss 