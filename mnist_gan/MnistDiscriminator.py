from torch import nn


class MnistDiscriminator(nn.Module):

  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(784, 1024),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(1024, 512),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = x.view(x.size(0), 784)
    return self.model(x)

