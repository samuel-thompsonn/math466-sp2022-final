from torch import nn


class Discriminator(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(2, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(64, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    return self.model(x)