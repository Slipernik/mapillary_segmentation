import torch
import torch.nn as nn

class SelfMadeUNet(nn.Module):
  """
  U-Net декодер поверх CNN-backbone (backbone.features).
  Скипы берутся из encoder-стадий backbone, апсемплинг — ConvTranspose2d.
  """

  def __init__(self, out_ch, backbone, layers=[6, 13, 26, 39]):
    """
    Args:
        out_ch (int): число выходных каналов (классов)
        backbone (nn.Module): модель с backbone.features
        layers (tuple[int,int,int,int]): индексы границ encoder-блоков/пулинга
        в features
    """
    super().__init__()
    self.backbone = backbone

    # Encoder (разрезаем backbone.features на блоки + pool-слои между ними)
    self.enc1 = self.backbone.features[:layers[0]]
    self.pool1 = self.backbone.features[layers[0]]
    self.enc2 = self.backbone.features[layers[0]+1:layers[1]]
    self.pool2 = self.backbone.features[layers[1]]
    self.enc3 = self.backbone.features[layers[1]+1:layers[2]]
    self.pool3 = self.backbone.features[layers[2]]
    self.enc4 = self.backbone.features[layers[2]+1:layers[3]]
    self.pool4 = self.backbone.features[layers[3]]

    self.bottleneck = self.backbone.features[40:52]

    # Decoder (апсемплинг + concat skip + conv_block)
    self.up_conv1 = nn.ConvTranspose2d(512, 512, 2, stride=2)
    self.conv1 = self.conv_block(1024, 512)
    self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
    self.conv2 = self.conv_block(512, 256)
    self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
    self.conv3 = self.conv_block(256, 128)
    self.up_conv4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
    self.conv4 = self.conv_block(128, 64)
    self.conv5 = nn.Conv2d(64, out_ch, 1)

  def forward(self, x):
    # Encoder + skip features
    x1 = self.enc1(x)
    out = self.pool1(x1)

    x2 = self.enc2(out)
    out = self.pool2(x2)

    x3 = self.enc3(out)
    out = self.pool3(x3)

    x4 = self.enc4(out)
    out = self.pool4(x4)

    bn = self.bottleneck(out)

    # Decoder: up -> concat(skip) -> conv
    out = self.up_conv1(bn)
    x4 = torch.concat((x4, out), dim=1)
    out = self.conv1(x4)

    out = self.up_conv2(out)
    x3 = torch.concat((x3, out), dim=1)
    out = self.conv2(x3)

    out = self.up_conv3(out)
    x2 = torch.concat((x2, out), dim=1)
    out = self.conv3(x2)

    out = self.up_conv4(out)
    x1 = torch.concat((x1, out), dim=1)

    out = self.conv4(x1)
    out = self.conv5(out)

    return out

  def conv_block(self, in_ch, out_ch):
    """(Conv-BN-ReLU) x2"""
    return nn.Sequential(
      nn.Conv2d(in_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
      nn.Conv2d(out_ch, out_ch, 3, padding=1),
      nn.BatchNorm2d(out_ch),
      nn.ReLU(),
    )