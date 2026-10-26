# --- cell 1 ---
import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    n_samples, n_features = X.shape
    X_augmented = np.hstack((np.ones((n_samples, 1)), X))
    beta = np.linalg.pinv(X_augmented.T @ X_augmented) @ X_augmented.T @ y
    trained_bias = float(beta[0])
    trained_weights = beta[1:]
    return trained_weights, trained_bias