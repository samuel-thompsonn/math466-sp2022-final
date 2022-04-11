import torch

def create_real_labels(size, device):
  data = torch.ones(size, 1)
  return data.to(device)

def create_fake_labels(size, device):
  data = torch.zeros(size, 1)
  return data.to(device)
  
def create_noise(sample_size, latent_space_size, device):
  return torch.randn(sample_size, latent_space_size).to(device)
