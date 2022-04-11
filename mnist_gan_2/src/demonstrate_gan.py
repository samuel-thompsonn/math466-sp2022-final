import argparse
import matplotlib.pyplot as plt
from Generator import Generator
import torch
from gan_util import create_noise

LATENT_VECTOR_SIZE = 128

def demonstrate_gan(generator_path):
  device = torch.device('cpu')
  generator = Generator(LATENT_VECTOR_SIZE, device)
  generator.load_state_dict(torch.load(generator_path))
  generated_samples = generator(create_noise(16, LATENT_VECTOR_SIZE, device)).detach()
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
  plt.show()

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("generator_filepath")
  args = parser.parse_args()
  demonstrate_gan(args.generator_filepath)
