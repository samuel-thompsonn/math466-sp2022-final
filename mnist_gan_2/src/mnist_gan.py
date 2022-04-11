from gan_util import create_noise
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os

from Generator import Generator
from Discriminator import Discriminator

from tqdm import tqdm
import argparse

BATCH_SIZE = 512
SAMPLE_SIZE = 64
LATENT_VECTOR_SIZE = 128
NUM_DISCRIMINATOR_STEPS = 1
LEARNING_RATE = 0.0002

def _get_training_data():
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,)),
  ])

  to_pil_image = transforms.ToPILImage()

  training_data = datasets.MNIST(
    root="../input/data",
    train=True,
    download=True,
    transform=transform
  )
  return training_data

def save_generator_image(image, path):
  save_image(image, path)

def save_gan(output_folder, generator, discriminator, epoch):
  print("Saving generator and discriminator...")
  if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
  if not os.path.isdir(f"{output_folder}/epoch_{epoch}"):
    os.mkdir(f"{output_folder}/epoch_{epoch}")
  torch.save(generator.to(device=torch.device('cpu')).state_dict(), f"{output_folder}/epoch_{epoch}/generator.dat")
  torch.save(discriminator.state_dict(), f"{output_folder}/epoch_{epoch}/discriminator.dat")
  print("Generator and discriminator saved.")
  if torch.cuda.is_available():
    generator.to(device=torch.device('cuda'))

def train_gan(num_epochs, batch_size, start_epoch=0, generator_path=None, discriminator_path=None): 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  training_data = _get_training_data()
  batch_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
  generator = Generator(LATENT_VECTOR_SIZE, device)
  if generator_path is not None:
    generator.load_state_dict(torch.load(generator_path))
  generator = generator.to(device)
  discriminator = Discriminator(device).to(device)
  if discriminator_path is not None:
    discriminator.load_state_dict(torch.load(discriminator_path))

  optimizer_generator = optim.Adam(generator.parameters(), LEARNING_RATE)
  optimizer_discriminator = optim.Adam(discriminator.parameters(), LEARNING_RATE)

  criterion = nn.BCELoss()

  losses_generator = []
  losses_discriminator = []
  images = []

  generator.train()
  discriminator.train()

  for epoch in range(start_epoch, num_epochs+start_epoch):
    loss_generator = 0.0
    loss_discriminator = 0.0
    total_batches = int(len(training_data)/batch_loader.batch_size)
    for batch_index, data in tqdm(enumerate(batch_loader), total=total_batches):
      image, _ = data
      image = image.to(device)
      num_data_points = len(image)
      for _ in range(NUM_DISCRIMINATOR_STEPS):
        generated_data = generator(create_noise(num_data_points, LATENT_VECTOR_SIZE, device))
        real_data = image
        loss_discriminator += discriminator.do_training_step(
          optimizer_discriminator,
          real_data,
          generated_data,
          criterion
        )
      loss_generator += generator.do_training_step(optimizer_generator, discriminator, batch_size, criterion)
    epoch_loss_generator = loss_generator / total_batches
    epoch_loss_discriminator = loss_discriminator / total_batches

    print(f"Epoch {epoch} of {num_epochs}:")
    print(f"Generator loss: {epoch_loss_generator:.8f}")
    print(f"Discriminator loss: {epoch_loss_discriminator:.8f}")
    save_gan("../outputs/saved_gans", generator, discriminator, epoch)


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("num_epochs", type=int)
  parser.add_argument("-generator_path")
  parser.add_argument("-discriminator_path")
  parser.add_argument("-batch_size", type=int, default=BATCH_SIZE)
  parser.add_argument("-start_epoch", type=int, default=0)
  args = parser.parse_args()

  train_gan(args.num_epochs, args.batch_size, start_epoch=args.start_epoch,
    discriminator_path=args.discriminator_path,
    generator_path=args.generator_path
  )
