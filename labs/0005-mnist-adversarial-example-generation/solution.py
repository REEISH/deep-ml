# --- cell 1 ---
import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x.
    
    Args:
        model: Pre-trained classifier (already in eval mode)
        x: Input image tensor, shape (1, 1, 28, 28), values in [0,1]
        y: True label, shape (1,) or scalar
        epsilon: L∞ perturbation budget
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
    
    Returns:
        x_adv: Adversarial example, same shape as x, satisfying:
               - ||x_adv - x||_∞ ≤ epsilon
               - x_adv values in [0, 1]
               - model(x_adv).argmax() != y (ideally)
    '''
    # TODO: Implement your adversarial attack here
    # Hint: Use gradients of loss w.r.t. input x
    if epsilon <= 0:
        return x.clone().detach()
    target = y.view(1) if y.dim() == 0 else y
    num_steps = 40
    alpha = epsilon / 10.0
    delta = torch.empty_like(x).uniform_(-epsilon, epsilon)
    x_adv = torch.clamp(x + delta, min=0.0, max=1.0).detach()
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        outputs = model(x_adv)
        loss = criterion(outputs, target)
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            grad_sign = x_adv.grad.sign()
            x_adv = x_adv + alpha * grad_sign
            delta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(x + delta, min=0.0, max=1.0)
    return x_adv.detach()