import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dcw_autoencoder import DcwAutoencoder
from torchvision import transforms
from animal_faces import AnimalFaces
from dummy_dataset import DummyDataset


def _freeze_except_batchnorms(model):
  n = 0
  for child in model.children():
    _freeze_except_batchnorms(child)
    n += 1

  if n == 0:
    for param in model.parameters():
      param.requires_grad = \
        isinstance(model, torch.nn.BatchNorm2d)


def _create_dataloaders(use_dummy, dataset_location,
                        batch_size, num_workers):
  # https://pytorch.org/hub/pytorch_vision_resnet/
  preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(
      mean=[0.485, 0.456, 0.406],
      std=[0.229, 0.224, 0.225]),
  ])

  datasets = []
  for train in [True, False]:
    if use_dummy:
      datasets.append(
        DummyDataset(
          num=batch_size * 2,
          shape=(512, 512, 3),
          transform=preprocess,
        ),
      )
    else:
      datasets.append(
        AnimalFaces(
          dataset_location,
          train=train,
          img_transform=preprocess,
          shuffle=True,
          with_label=False,
        ),
      )

  return [torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, num_workers=num_workers
  ) for dataset in datasets]


class Training_autoencoder(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = DcwAutoencoder()

    self.dataset_location = None
    self.num_workers = 0
    self.test_run = False

    self.batch_size = 64
    self.max_epochs = 100
    self.lr = 0.001
    self.lr_decay = None

    self.gpus = None
    self.fine_tune = False
    self.ts_log_dir = None
    self.model_save_dir = None
    self.experiment_name = None
    self.version = None
  
  def forward(self, x):
    return self.model(x)
  
  def training_step(self, batch, step):
    return self._step(batch, step, phase='train')
  
  def validation_step(self, batch, step):
    return self._step(batch, step, phase='val')
  
  def training_epoch_end(self, results):
    return self._step_end(results, phase='train')

  def validation_epoch_end(self, results):
    return self._step_end(results, phase='val')

  def _step_end(self, results, phase):
    losses = []
    for result in results:
      log = result['log']
      losses.append(log['%s/loss' % phase])
    avg_loss = torch.stack(losses).mean()
    log = {'%s/loss' % phase: avg_loss}
    loss_key = 'loss' if phase == 'train' else 'val_loss'
    return {loss_key: avg_loss, 'log': log}

  def _step(self, batch, step, phase):
    imgs = batch
    _, reconstructed = self.model._forward(imgs)
    loss = F.mse_loss(imgs, reconstructed)
    log = {'%s/loss' % phase: loss}
    loss_key = 'loss' if phase == 'train' else 'val_loss'
    return {loss_key: loss, 'log': log}

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
    if self.lr_decay is not None:
      scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_decay)
      return [optimizer], [scheduler]
    return optimizer
  
  def start(self):
    assert self.experiment_name
    assert self.test_run or self.dataset_location
    assert self.ts_log_dir

    if not self.fine_tune:  # transfer learning
      _freeze_except_batchnorms(self.model)
    
    logger = TensorBoardLogger(
      save_dir=self.ts_log_dir,
      name=self.experiment_name,
      version=self.version)
    
    checkpoint_callback = None
    if self.model_save_dir:
      checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filepath='%s/{epoch}-{val_loss:.2f}' % self.model_save_dir,
        mode='min')

    trainer = pl.Trainer(
      logger=logger,
      gpus=self.gpus,
      max_epochs=self.max_epochs,
      checkpoint_callback=checkpoint_callback)

    train_ds, val_ds = _create_dataloaders(
      use_dummy=self.test_run,
      dataset_location=self.dataset_location,
      batch_size=self.batch_size,
      num_workers=self.num_workers)

    trainer.fit(
      self,
      train_dataloader=train_ds,
      val_dataloaders=val_ds)
