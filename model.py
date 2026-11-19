"""Independent implementation of the CircuitNet GPDL FCN architecture.

Architecture reference:
https://github.com/circuitnet/CircuitNet/blob/main/routability_ir_drop_prediction/models/gpdl.py
One output channel matches the released congestion preprocessing script.
"""
import torch
from torch import nn
from torch.nn import functional as F


def block(a, b, *, second_dilation=1):
    layers = []
    for cin, dilation in ((a, 1), (b, second_dilation)):
        layers.extend([nn.Conv2d(cin, b, 3, padding=dilation, dilation=dilation),
                       nn.InstanceNorm2d(b, affine=True), nn.LeakyReLU(.2)])
    return nn.Sequential(*layers)


def up(a, b):
    return nn.Sequential(nn.ConvTranspose2d(a, b, 4, 2, 1),
                         nn.InstanceNorm2d(b, affine=True), nn.LeakyReLU(.2))


def initialize_conv(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0., .02)
        nn.init.zeros_(module.bias)


class FCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1, self.enc2 = block(3, 32), block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1),
                                        nn.BatchNorm2d(32), nn.Tanh())
        self.dec1, self.up1 = block(32, 32), up(32, 16)
        self.dec2, self.up2 = block(16, 16), up(48, 4)
        self.head = nn.Sequential(nn.Conv2d(4, 1, 3, padding=1), nn.Sigmoid())
        self.apply(initialize_conv)

    def forward(self, x):
        if x.shape[-1] % 4 or x.shape[-2] % 4:
            raise ValueError('Spatial dimensions must be divisible by four')
        skip = self.pool(self.enc1(x))
        h = self.bottleneck(self.pool(self.enc2(skip)))
        h = self.dec2(self.up1(self.dec1(h)))
        return self.head(self.up2(torch.cat((h, skip), dim=1)))


class UNet(nn.Module):
    """Three skip resolutions with a parameter budget close to the FCN."""

    def __init__(self, bottleneck_dilation=1):
        super().__init__()
        self.enc1 = block(3, 12)
        self.enc2 = block(12, 24)
        self.enc3 = block(24, 32)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = block(32, 48, second_dilation=bottleneck_dilation)
        self.up3 = nn.ConvTranspose2d(48, 32, 2, stride=2)
        self.dec3 = block(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 24, 2, stride=2)
        self.dec2 = block(48, 24)
        self.up1 = nn.ConvTranspose2d(24, 12, 2, stride=2)
        self.dec1 = block(24, 12)
        self.head = nn.Sequential(nn.Conv2d(12, 1, 1), nn.Sigmoid())
        self.apply(initialize_conv)

    def decode(self, x):
        if x.shape[-1] % 8 or x.shape[-2] % 8:
            raise ValueError('Spatial dimensions must be divisible by eight')
        skip1 = self.enc1(x)
        skip2 = self.enc2(self.pool(skip1))
        skip3 = self.enc3(self.pool(skip2))
        h = self.bottleneck(self.pool(skip3))
        h = self.dec3(torch.cat((self.up3(h), skip3), dim=1))
        h = self.dec2(torch.cat((self.up2(h), skip2), dim=1))
        return self.dec1(torch.cat((self.up1(h), skip1), dim=1))

    def forward(self, x):
        return self.head(self.decode(x))


class HotspotUNet(UNet):
    """Predict congestion values and hotspot logits from the shared decoder."""

    def __init__(self, bottleneck_dilation=1):
        super().__init__(bottleneck_dilation=bottleneck_dilation)
        self.hotspot_head = nn.Conv2d(12, 1, 1)
        initialize_conv(self.hotspot_head)

    def forward(self, x):
        features = self.decode(x)
        return self.head(features), self.hotspot_head(features)


class ContextHotspotUNet(HotspotUNet):
    """Condition both output heads on the surrounding placement demand."""

    def __init__(self, bottleneck_dilation=1):
        super().__init__(bottleneck_dilation=bottleneck_dilation)
        # Six summaries per input channel. The last layer starts at identity
        # modulation so the original two heads have the same initial predictions.
        self.context = nn.Sequential(nn.Conv2d(18, 24, 3, padding=1), nn.LeakyReLU(.2),
                                     nn.Conv2d(24, 4, 1))
        nn.init.zeros_(self.context[-1].weight)
        nn.init.zeros_(self.context[-1].bias)

    def forward(self, x):
        features = self.decode(x)
        local = F.avg_pool2d(x, 8)
        height, width = local.shape[-2:]
        regions = F.adaptive_avg_pool2d(local, (min(4, height), min(4, width)))
        regions = F.interpolate(regions, size=(height, width), mode='bilinear', align_corners=False)
        deviation, mean = torch.std_mean(x, dim=(-2, -1), correction=0, keepdim=True)
        summaries = (local, regions, local.mean(-1, keepdim=True).expand_as(local),
                     local.mean(-2, keepdim=True).expand_as(local), mean.expand_as(local),
                     deviation.expand_as(local))
        modulation = self.context(torch.cat(summaries, dim=1))
        modulation = F.interpolate(modulation, size=x.shape[-2:], mode='bilinear', align_corners=False)
        map_scale, map_bias, hotspot_scale, hotspot_bias = modulation.chunk(4, dim=1)
        congestion = (1 + map_scale.tanh()) * self.head[0](features) + map_bias
        logits = (1 + hotspot_scale.tanh()) * self.hotspot_head(features) + hotspot_bias
        return congestion.sigmoid(), logits
