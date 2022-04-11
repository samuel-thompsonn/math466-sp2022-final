from MnistDiscriminator import MnistDiscriminator
from MnistGenerator import MnistGenerator
import torch
import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import argparse
import os

SEED = 111
TRAINING_DATA_LENGTH = 1024
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 50
LOSS_FUNCTION = torch.nn.BCELoss()

def _generate_training_data(transform):
  return torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
  )

def _get_batch_loader(batch_size, training_set):
  return torch.utils.data.DataLoader(
    training_set, batch_size=batch_size, shuffle=True
  )

def _generate_discriminator_training_data(
  real_samples_labels,
  real_samples,
  batch_size,
  generator,
  device
):
  real_samples = real_samples.to(device=device)
  latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
  generated_samples = generator(latent_space_samples)
  generated_sample_labels = torch.zeros((batch_size, 1)).to(device=device)
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
  batch_size,
  device
):
  return torch.randn((batch_size, 100)).to(device=device)

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

def save_gan(output_folder, generator, discriminator, epoch):
  print("Saving generator and discriminator...")
  if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
  if not os.path.isdir(f"{output_folder}/epoch_{epoch}"):
    os.mkdir(f"{output_folder}/epoch_{epoch}")
  torch.save(generator.to(device=torch.device('cpu')).state_dict(), f"{output_folder}/epoch_{epoch}/generator.dat")
  torch.save(discriminator.state_dict(), f"{output_folder}/epoch_{epoch}/discriminator.dat")
  print("Generator and discriminator saved.")

def run_gan(discriminator_path=None, generator_path=None, num_epochs=NUM_EPOCHS, output_folder="data"):

  torch.manual_seed(SEED)

  device = torch.device("cpu")
  if torch.cuda.is_available():
    device = torch.device("cuda")

  transform = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))
  ])

  training_set = _generate_training_data(transform)
  batch_loader = _get_batch_loader(BATCH_SIZE, training_set)

  discriminator = MnistDiscriminator()
  if discriminator_path is not None:
    discriminator.load_state_dict(torch.load(discriminator_path))
  discriminator = discriminator.to(device)

  generator = MnistGenerator()
  if generator_path is not None:
    generator.load_state_dict(torch.load(generator_path))
  generator = generator.to(device)

  optimizier_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
  optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)

  for epoch in range(num_epochs):
    loss_discriminator = None
    loss_generator = None
    for n, (real_samples, mnist_labels) in enumerate(batch_loader):
      real_samples_labels = torch.ones((BATCH_SIZE, 1)).to(device=device)
      all_samples, all_sample_labels = _generate_discriminator_training_data(
        real_samples_labels,
        real_samples,
        BATCH_SIZE,
        generator,
        device
      )
      loss_discriminator = _train_discriminator(
        discriminator,
        LOSS_FUNCTION,
        all_samples,
        all_sample_labels,
        optimizier_discriminator
      )
      latent_space_samples = _generate_generator_training_data(BATCH_SIZE, device)
      loss_generator = _train_generator(
        generator,
        latent_space_samples,
        discriminator,
        LOSS_FUNCTION,
        real_samples_labels,
        optimizer_generator
      )
    print(f"Epoch: {epoch} Loss Discriminator: {loss_discriminator}")
    print(f"Epoch: {epoch} Loss Generator: {loss_generator}")
    save_gan(output_folder, generator, discriminator, epoch)
  save_gan(output_folder, generator, discriminator, num_epochs)
  print(f"{num_epochs} epochs of training complete. Exiting...")

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("num_epochs", type=int)
  parser.add_argument("output_folder", help="Filepath of directory to output discriminator and generator dicts", default="data")
  parser.add_argument("-discriminator", help="Filepath of discriminator state dict", default=None)
  parser.add_argument("-generator", help="Filepath of generator state dict", default=None)
  args = parser.parse_args()
  run_gan(
    generator_path=args.generator,
    discriminator_path=args.discriminator,
    num_epochs=args.num_epochs,
    output_folder=args.output_folder
  )
