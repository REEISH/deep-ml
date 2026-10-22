# --- cell 1 ---
import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    if 't' not in state:
        state['t'] = 0
        state['m'] = np.zeros_like(param)
        state['v'] = np.zeros_like(param)
    state['t'] += 1
    t = state['t']
    state['m'] = beta1*state['m'] + (1 - beta1)*grad
    state['v'] = beta2*state['v'] + (1 - beta1)*(grad**2)
    m_hat = state['m']/(1 - beta1**state['t'])
    v_hat = state['v']/(1 - beta1**state['t'])
    new_param = param - lr*(m_hat/(np.sqrt(v_hat) + eps))  # Replace with your update rule
    return new_param, state