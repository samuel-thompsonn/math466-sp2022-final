import argparse
import matplotlib.pyplot as plt
from MnistGenerator import MnistGenerator
import torch

BATCH_SIZE = 32

def demonstrate_gan(generator_path):
  device = torch.device("cpu")
  if torch.cuda.is_available():
    device = torch.device("cuda")
  generator = MnistGenerator().to(device=device)
  generator.load_state_dict(torch.load(generator_path))
  latent_space_samples = torch.randn(BATCH_SIZE, 100).to(device=device)
  generated_samples = generator(latent_space_samples).cpu().detach()
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
