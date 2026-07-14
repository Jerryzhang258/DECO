import torch
import einops
from torch import nn
from models.deco.img_encoder import BasicBlock


class TactileResNet(nn.Module):
    """Lightweight (resnet18-lite) shared encoder for vision-based tactile RGB images.

    Official ManiSkill-ViTac tactile streams are camera-like 224x224 RGB images from
    vision-based tactile sensors (e.g. DuoTact/AllTact/GelSight), not force scalars, so a
    small conv backbone is used instead of DECO's original 1062-dim region encoder.
    """

    def __init__(self, out_channels=256):
        super().__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(32, 2, stride=1)
        self.layer2 = self._make_layer(64, 2, stride=2)
        self.layer3 = self._make_layer(128, 2, stride=2)
        self.layer4 = self._make_layer(out_channels, 2, stride=2)

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class TactileImageEncoder(nn.Module):
    """Shared encoder for the 4 vision-based tactile streams (tactile_left_0, tactile_right_0,
    tactile_left_1, tactile_right_1 -- "_0"/"_1" index which robot arm, "left"/"right" index
    which of the two sensor pads on that gripper), optionally repeated over `t_hist` timesteps.

    One shared backbone is used for all sensors/timesteps (mirrors how DECO shares one ResNet-34
    across the two wrist cameras). Output is a set of spatial tokens per sensor rather than a
    single pooled vector, so the tactile cross-attention can still localize contact within the
    sensor pad.
    """

    def __init__(self, dim=512, pool_size=2, t_hist=1, n_sensors=4):
        super().__init__()
        self.pool_size = pool_size
        self.t_hist = t_hist
        self.n_sensors = n_sensors

        self.encoder = TactileResNet(out_channels=256)
        self.head = nn.Conv2d(256, dim, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        # physical sensor identity: e.g. left-arm/left-pad, left-arm/right-pad, right-arm/left-pad, right-arm/right-pad
        self.sensor_id_embedd = nn.Embedding(n_sensors, dim)
        if t_hist > 1:
            self.time_id_embedd = nn.Embedding(t_hist, dim)

    def forward(self, tactile_imgs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tactile_imgs: [B, n_sensors * t_hist, 3, H, W]. Ordered as all sensors for the
                oldest frame first, then all sensors for the next frame, etc.
        Returns:
            tokens: [B, n_sensors * t_hist * pool_size**2, dim]
        """
        b, n, c, h, w = tactile_imgs.shape
        x = tactile_imgs.reshape(b * n, c, h, w)
        x = self.encoder(x)
        x = self.head(x)
        x = self.pool(x)  # (b*n, dim, pool_size, pool_size)
        x = einops.rearrange(x, "(b n) c ph pw -> b n (ph pw) c", b=b, n=n)

        sensor_ids = torch.arange(self.n_sensors, device=x.device).repeat(self.t_hist)
        x = x + self.sensor_id_embedd(sensor_ids)[None, :, None, :]

        if self.t_hist > 1:
            time_ids = torch.arange(self.t_hist, device=x.device).repeat_interleave(self.n_sensors)
            x = x + self.time_id_embedd(time_ids)[None, :, None, :]

        x = einops.rearrange(x, "b n l c -> b (n l) c")
        return x
