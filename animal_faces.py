"""
Download the dataset from here:
https://www.kaggle.com/andrewmvd/animal-faces
"""

import torch
from PIL import Image
from pathlib import Path
import random


_LABELS = {
  'cat': 0,
  'dog': 1,
  'wild': 2,
}


class AnimalFaces(torch.utils.data.Dataset):
  def __init__(self, dataset_location, label=None, train=True,
               img_transform=None, label_transform=None,
               shuffle=True, with_label=True, integer_label=True):
    super().__init__()
    self.img_transform = img_transform
    self.label_trainsform = label_transform
    self.with_label = with_label
    self.integer_label = integer_label
    self._paths = []
    section = 'train' if train else 'val'
    if not label:
      labels = list(_LABELS.keys())
    for label in labels:
      p = '%s/%s/%s' % (dataset_location, section, label)
      paths = [str(p) for p in Path(p).rglob('*')]
      self._paths.extend(paths)
    if shuffle:
      random.shuffle(self._paths)
  
  def __len__(self):
    return len(self._paths)
  
  def __getitem__(self, index):
    path = self._paths[index]
    img = Image.open(path)
    if self.img_transform:
      img = self.img_transform(img)
    if not self.with_label:
      return img
    label = path.split('/')[-2]
    if self.integer_label:
      label = _LABELS[label]
    if self.label_trainsform:
      label = self.label_trainsform(label)
    return img, label
