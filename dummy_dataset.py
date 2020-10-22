import torch
import numpy as np
from PIL import Image


class DummyDataset(torch.utils.data.Dataset):
  def __init__(self, num, shape, transform):
    self.num = num
    self.shape = shape
    self.transform = transform

  def __len__(self):
    return self.num
  
  def __getitem__(self, index):
    img = np.random.randint(0, 256, self.shape).astype(np.uint8)
    img = Image.fromarray(img)
    if self.transform:
      return self.transform(img)
    return img
