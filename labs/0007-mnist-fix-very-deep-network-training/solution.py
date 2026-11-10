# --- cell 1 ---
import torch
import torch.nn as nn

class DeepNetwork(nn.Module):
    '''
    A very deep network that trains poorly!
    
    This 30-layer network achieves only ~20% accuracy due to:
    - Vanishing gradients
    - No skip connections
    - No normalization
    - Poor gradient flow
    
    Your task: Fix this network to achieve 90%+ accuracy while keeping ≥30 layers!
    
    You can:
    - Add skip/residual connections (ResNet style)
    - Add dense connections (DenseNet style)
    - Add batch normalization or layer normalization
    - Change activation functions
    - Reorganize the architecture
    - Add any architectural improvements you know
    
    You CANNOT:
    - Reduce the number of layers below 30
    '''
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        
        # Input projection
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.input_bn = nn.BatchNorm1d(hidden_size)
        
        # 30 hidden layers (this is the deep part that needs fixing!)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(30)
        ])
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_size) for _ in range(30)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, num_classes)
        
        # Activation
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        
        # Input projection
        x = self.activation(self.input_bn(self.input_layer(x)))
        
        # Pass through 30 layers (problematic without skip connections!)
        for layer, bn in zip(self.layers, self.bns):
            residual = x
            out = layer(x)
            out = bn(out)
            out = self.activation(out)
            x = out + residual
        
        # Output
        x = self.output_layer(x)
        
        return x