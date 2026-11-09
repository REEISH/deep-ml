# --- cell 1 ---
import numpy as np

def train(X_train, y_train, X_val, y_val):
    """
    Train a regression model that generalizes well despite having
    MORE features than training samples (many are noise or redundant).
    
    WARNING: An unregularized approach WILL overfit here.
    - Unregularized OLS: Train R² ≈ 1.0, Val R² ≈ -5.0
    - You need regularization to pass!
    
    Args:
        X_train: numpy array of shape (n_samples, n_features) -- standardized
                 (~250 samples, ~264 features -- more features than samples!)
        y_train: numpy array of shape (n_samples,) -- target values
        X_val:   numpy array of shape (n_val, n_features) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation targets
    
    Returns:
        predict: callable that takes X (n, n_features) and returns y_pred (n,)
    """
    # TODO: implement a training strategy that avoids overfitting
    # Some ideas:
    #   - Ridge regression (L2): add alpha * ||W||^2 to loss
    #   - Gradient descent with L2 penalty
    #   - Feature selection (remove low-variance or uncorrelated features)
    #   - Early stopping on gradient descent
    #   - Any combination of the above!
    n_samples, n_features = X_train.shape
    
    y_train_mean = np.mean(y_train)
    y_train_centered = y_train - y_train_mean
    
    y_val_mean = np.mean(y_val)
    ss_tot_val = np.sum((y_val - y_val_mean) ** 2)
    
    # 1. Feature Selection: score features by absolute correlation with the target
    corrs = np.abs(X_train.T @ y_train_centered)
    
    best_r2 = -np.inf
    best_w = None
    best_top_k_idx = None
    
    # Test different subsets of the most correlated features
    K_list = [5, 10, 20, 30, 50, 100, 150, 200, n_features]
    K_list = sorted(list(set([k for k in K_list if k <= n_features])))
    
    # Expand alpha range for heavier regularization
    alphas = np.logspace(-2, 6, 50)
    
    for k in K_list:
        # Keep only the top 'k' features
        top_k_idx = np.argsort(corrs)[-k:]
        
        X_train_k = X_train[:, top_k_idx]
        X_val_k = X_val[:, top_k_idx]
        
        # 2. Stable Ridge Regression via Singular Value Decomposition (SVD)
        # Computing X^T X is catastrophic for polynomials. SVD calculates 
        # the exact Ridge solution without squaring the condition number.
        U, S, Vt = np.linalg.svd(X_train_k, full_matrices=False)
        UTy = U.T @ y_train_centered
        
        for alpha in alphas:
            # Ridge solution: w = V * diag(S / (S^2 + alpha)) * U^T * y
            d = S / (S**2 + alpha)
            w_k = Vt.T @ (d * UTy)
            
            y_val_pred = (X_val_k @ w_k) + y_train_mean
            
            ss_res_val = np.sum((y_val - y_val_pred) ** 2)
            r2_val = 1.0 - (ss_res_val / ss_tot_val) if ss_tot_val > 0 else -np.inf
            
            if r2_val > best_r2:
                best_r2 = r2_val
                best_w = w_k
                best_top_k_idx = top_k_idx
                
    if best_w is None:
        best_w = np.zeros(1)
        best_top_k_idx = np.array([0])
        
    # Safely cap predictions to the training distribution 
    y_min, y_max = np.min(y_train), np.max(y_train)
        
    def predict(X):
        X = np.asarray(X)
        X_k = X[:, best_top_k_idx]
        preds = (X_k @ best_w) + y_train_mean
        
        # 3. Clip predictions to prevent infinite errors on outlier test features
        return np.clip(preds, y_min, y_max)

    return predict