from torch import nn
import torchvision.models as models


class _ConvTrans(nn.Module):
  def __init__(self, in_channels, out_channels, activation,
               kernel_size=2, stride=2, padding=0, momentum=0.01):
    super(_ConvTrans, self).__init__()
    self.conv = nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size, stride, padding)
    self.norm = nn.BatchNorm2d(out_channels, momentum=momentum)
    self.activation = activation

  def forward(self, x):
    x = self.conv(x)
    x = self.norm(x)
    x = self.activation(x)
    return x


class _PointWiseConv(nn.Conv2d):
  def __init__(self, in_channels, out_channels):
    super(_PointWiseConv, self).__init__(
      in_channels, out_channels,
      kernel_size=1, stride=1, padding=0)


class DcwAutoencoder(nn.Module):
  def __init__(self):
    super(DcwAutoencoder, self).__init__()

    resnet = models.resnet50(pretrained=True)
    modules = list(resnet.children())[:-2]  # use the last bottleneck
    resnet = nn.Sequential(*modules)

    self.encoder = resnet
    self.decoder = nn.Sequential(
      _ConvTrans(2048, 1024, activation=nn.ReLU()),  # 1024x14x14
      _ConvTrans(1024, 512, activation=nn.ReLU()),  # 512x28x28
      _ConvTrans(512, 256, activation=nn.ReLU()),  # 256x56x56
      _ConvTrans(256, 128, activation=nn.ReLU()),  # 128x112x112
      _ConvTrans(128, 64, activation=nn.ReLU()),  # 128x224x224
      _PointWiseConv(64, 3),  # 3x224x224
      nn.Sigmoid(),
    )
  
  def forward(self, x):
    embedding, _ = self._forward(x)
    return embedding
  
  def _forward(self, x):
    embedding = self.encoder(x)
    reconstructed = self.decoder(embedding)
    return embedding, reconstructed
