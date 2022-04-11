import argparse
import matplotlib.pyplot as plt
from Generator import Generator
import torch

def demonstrate_gan(generator_path):
  generator = Generator()
  generator.load_state_dict(torch.load(generator_path))
  latent_space_samples = torch.randn(100, 2)
  generated_samples = generator(latent_space_samples).detach()
  plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
  plt.show()

if __name__=='__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("generator_filepath")
  args = parser.parse_args()
  demonstrate_gan(args.generator_filepath)
