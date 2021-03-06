import string
import torch
from Discriminator import Discriminator
from Generator import Generator
import math
import matplotlib.pyplot as plt
import argparse
import os

SEED = 111
TRAINING_DATA_LENGTH = 1024
BATCH_SIZE = 32
LR = 0.001
NUM_EPOCHS = 300
LOSS_FUNCTION = torch.nn.BCELoss()

def _generate_training_data(num_points):
  train_data = torch.zeros((num_points, 2))
  train_data[:, 0] = 2 * math.pi * torch.rand(num_points)
  train_data[:, 1] = torch.sin(train_data[:, 0])
  train_labels = torch.zeros(num_points)
  train_set = [
    (train_data[i], train_labels[i]) for i in range(num_points)
  ]
  return train_set

def _get_batch_loader(batch_size, training_set):
  return torch.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True
  )

def _generate_discriminator_training_data(
  real_samples_labels,
  real_samples,
  batch_size,
  generator
):
  latent_space_samples = torch.randn((batch_size, 2))
  generated_samples = generator(latent_space_samples)
  generated_sample_labels = torch.zeros((batch_size, 1))
  all_samples = torch.cat((real_samples, generated_samples))
  all_sample_labels = torch.cat(
    (real_samples_labels, generated_sample_labels)
  )
  return all_samples, all_sample_labels

def _train_discriminator(
  discriminator,
  loss_function,
  all_samples,
  all_sample_labels,
  optimizer_discriminator
):
  discriminator.zero_grad()
  discriminator_output = discriminator(all_samples)
  discriminator_loss = loss_function(
    discriminator_output, all_sample_labels
  )
  discriminator_loss.backward()
  optimizer_discriminator.step()
  return discriminator_loss


def _generate_generator_training_data(
  batch_size
):
  return torch.randn((batch_size, 2))

def _train_generator(
  generator,
  latent_space_samples,
  discriminator,
  loss_function,
  real_sample_labels,
  optimizer_generator
):
  generator.zero_grad()
  generated_samples = generator(latent_space_samples)
  output_discriminator_generated = discriminator(generated_samples)
  loss_generator = loss_function(
    output_discriminator_generated, real_sample_labels
  )
  loss_generator.backward()
  optimizer_generator.step()
  return loss_generator

def _generate_samples(generator):
  latent_space_samples = torch.randn(100, 2)
  return generator(latent_space_samples).detach()

def run_gan(discriminator_path=None, generator_path=None, num_epochs=NUM_EPOCHS):

  generated_data_sets = []

  torch.manual_seed(SEED)
  training_set = _generate_training_data(TRAINING_DATA_LENGTH)
  batch_loader = _get_batch_loader(BATCH_SIZE, training_set)
  discriminator = Discriminator()
  if discriminator_path is not None:
    discriminator.load_state_dict(torch.load(discriminator_path))
  optimizier_discriminator = torch.optim.Adam(discriminator.parameters())
  generator = Generator()
  if generator_path is not None:
    generator.load_state_dict(torch.load(generator_path))
  optimizer_generator = torch.optim.Adam(generator.parameters())

  for epoch in range(num_epochs):
    for n, (real_samples, _) in enumerate(batch_loader):
      real_samples_labels = torch.ones((BATCH_SIZE, 1))
      all_samples, all_sample_labels = _generate_discriminator_training_data(
        real_samples_labels,
        real_samples,
        BATCH_SIZE,
        generator
      )
      loss_discriminator = _train_discriminator(
        discriminator,
        LOSS_FUNCTION,
        all_samples,
        all_sample_labels,
        optimizier_discriminator
      )
      latent_space_samples = _generate_generator_training_data(BATCH_SIZE)
      loss_generator = _train_generator(
        generator,
        latent_space_samples,
        discriminator,
        LOSS_FUNCTION,
        real_samples_labels,
        optimizer_generator
      )

      if epoch % 10 == 0 and n == BATCH_SIZE - 1:
          print(f"Epoch: {epoch} Loss Discriminator: {loss_discriminator}")
          print(f"Epoch: {epoch} Loss Generator: {loss_generator}")
  
  print(f"{num_epochs} epochs of training complete. Saving generator and discriminator...")
  if not os.path.isdir("data"):
    os.mkdir("data")
  torch.save(generator.state_dict(), "data/generator.dat")
  torch.save(discriminator.state_dict(), "data/discriminator.dat")
  print("Generator and discriminator saved. Exiting...")

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("num_epochs", type=int)
  parser.add_argument("-discriminator", help="Filepath of discriminator state dict", default=None)
  parser.add_argument("-generator", help="Filepath of generator state dict", default=None)
  args = parser.parse_args()
  run_gan(args.discriminator, args.generator, num_epochs=args.num_epochs)
