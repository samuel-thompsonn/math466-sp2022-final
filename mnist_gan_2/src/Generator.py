from gan_util import create_real_labels
from gan_util import create_noise
import torch.nn as nn

class Generator(nn.Module):
  
  def __init__(self, latent_space_dimensions, device):
    super(Generator, self).__init__()
    self.noise_dimensions = latent_space_dimensions
    self.device = device
    self.main = nn.Sequential(
      nn.Linear(self.noise_dimensions, 256),
      nn.LeakyReLU(0.2),
      nn.Linear(256, 512),
      nn.LeakyReLU(0.2),
      nn.Linear(512, 1024),
      nn.LeakyReLU(0.2),
      nn.Linear(1024, 784),
      nn.Tanh(),
    )
  
  def forward(self, x):
    return self.main(x).view(-1, 1, 28, 28)

  def do_training_step(self, optimizer, discriminator, batch_size, loss_function):
    fake_data = self(create_noise(batch_size, self.noise_dimensions, self.device))
    real_labels = create_real_labels(batch_size, self.device)
    optimizer.zero_grad()

    output = discriminator(fake_data)
    loss = loss_function(output, real_labels)

    loss.backward()
    optimizer.step()
    return loss
