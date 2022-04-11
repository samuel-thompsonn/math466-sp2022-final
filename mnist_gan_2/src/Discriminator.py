import torch.nn as nn

from gan_util import create_real_labels, create_fake_labels

class Discriminator(nn.Module):

  def __init__(self, device):
      super(Discriminator, self).__init__()
      self.n_input = 784
      self.device = device
      self.main = nn.Sequential(
        nn.Linear(self.n_input, 1024),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid(),
      )
  
  def forward(self, x):
    x = x.view(-1, 784)
    return self.main(x)

  def do_training_step(self, optimizer, real_data, generated_data, loss_function):
    num_real_data_points = real_data.size(0)
    real_labels = create_real_labels(num_real_data_points, self.device)
    fake_labels = create_fake_labels(num_real_data_points, self.device)

    optimizer.zero_grad()

    real_output = self(real_data)
    real_data_loss = loss_function(real_output, real_labels)

    fake_output = self(generated_data)
    fake_data_loss = loss_function(fake_output, fake_labels)

    real_data_loss.backward()
    fake_data_loss.backward()
    optimizer.step()

    return real_data_loss + fake_data_loss
