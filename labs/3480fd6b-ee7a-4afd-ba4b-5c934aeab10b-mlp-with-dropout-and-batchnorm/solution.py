# --- cell 1 ---
import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedMLP(nn.Module):
    """MLP with BatchNorm1d and Dropout for binary classification."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_p: float = 0.3):
        super().__init__()
        # TODO: Build at least two hidden blocks:
        #   Linear -> BatchNorm1d -> ReLU (or similar) -> Dropout
        # Final layer: Linear to 1 logit.
        # Store layers on self (Sequential is fine).
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) logits for batch x of shape (N, input_dim)."""
        # TODO: run x through your network and squeeze the last dim if needed
        logits = self.network(x)
        return logits.squeeze(-1)


def train_model(model, X_train, y_train, epochs=150, lr=1e-2):
    """Train model in-place with BCEWithLogitsLoss + Adam. Return model."""
    # TODO:
    # - criterion = nn.BCEWithLogitsLoss()
    # - optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # - loop epochs: zero_grad -> forward -> loss -> backward -> step
    # - y_train is float 0/1 with shape (N,)
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
    return model