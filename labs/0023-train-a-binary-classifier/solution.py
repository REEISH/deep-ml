# --- cell 1 ---
import numpy as np
# You can import any sklearn module you need

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    m, n = X_train.shape
    X_aug = np.hstack((np.ones((m, 1)), X_train))
    w_full = np.zeros(n + 1)
    learning_rate = 0.1
    epochs = 1000
    for epoch in range(epochs):
        logits = X_aug @ w_full
        preds = 1.0 / (1.0 + np.exp(-np.clip(logits, -500, 500)))
        errors = preds - y_train
        gradient = (X_aug.T @ errors)
        w_full -= learning_rate * gradient
        
    def predict(X: np.ndarray) -> np.ndarray:
        X_test_aug = np.hstack((np.ones((X.shape[0], 1)), X))
        test_logits = X_test_aug @ w_full
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(test_logits, -500, 500)))
        return np.where(probabilities >= 0.5, 1, 0)
    
    return predict