# --- cell 1 ---
import torch, torch.nn as nn

def build_model() -> nn.Module:
    """
    Return a tiny nn.Module for MNIST classification (10 classes).
    IMPORTANT: If total trainable params > 2048, final accuracy will be set to 0.
    Tip: Consider very small convs, global average pooling, and tiny linear head.
    """
    class TinyNet(nn.Module):
         def __init__(self):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.MaxPool2d(2) # 28x28 -> 14x14
            )
            
            
            self.block2 = nn.Sequential(
                nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.Conv2d(8, 12, kernel_size=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.MaxPool2d(2) # 14x14 -> 7x7
            )
            self.block3 = nn.Sequential(
                nn.Conv2d(12, 12, kernel_size=3, padding=1, groups=12),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.Conv2d(12, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            
            self.block4 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=1),
                nn.BatchNorm2d(16),
                nn.ReLU()
            )
            self.block5 = nn.Sequential(
                nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 20, kernel_size=1),
                nn.BatchNorm2d(20),
                nn.ReLU()
            )
            self.classifier_conv = nn.Conv2d(20, 10, kernel_size=1) 
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            
         def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = self.block3(x)
            x = self.block4(x)
            x = self.block5(x)
            x = self.classifier_conv(x)
            x = self.gap(x)
            return x.squeeze(-1).squeeze(-1)

             
    return TinyNet()