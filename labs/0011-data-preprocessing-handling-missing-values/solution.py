# --- cell 1 ---
import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    # TODO: Fill in NaN values
    n_features = X_clean.shape[1]
    for col_idx in range(n_features):
        column = X_clean[:, col_idx]
        nan_mask = np.isnan(column)
        if np.any(nan_mask):
            col_mean = np.nanmean(column)
            if np.isnan(col_mean):
                col_mean = 0.0
            column[nan_mask] = col_mean
    return X_clean