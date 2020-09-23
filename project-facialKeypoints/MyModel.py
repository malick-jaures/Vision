# Import deep learning resources from pytorch
from torch import nn
from torch.nn import functional as F

# Your model should accept 96x96 pixel graysale images in
# It should have a fully-connected output layer with 30 values (2 for each facial keypoint)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(2,2),
            nn.Dropout(0.2)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(64*12*12, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 30)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.regressor(x)
        
        return x
