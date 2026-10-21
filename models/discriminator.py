import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: list[int],
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        layers = []
        layers.extend([
            nn.Conv2d(
                in_channels=2*img_size[0],
                out_channels=in_channels[0],
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        for idx in range(len(in_channels) - 1):
            layers.extend([
                nn.Conv2d(
                    in_channels=in_channels[idx],
                    out_channels=in_channels[idx + 1],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.InstanceNorm2d(in_channels[idx + 1]),
                nn.LeakyReLU(0.2, inplace=True),
            ])
        layers.extend([
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(
                in_channels=in_channels[-1],
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=1,
            ),
        ])
        self.discriminator_layers = nn.Sequential(*layers)

    def forward(self, inputA: torch.Tensor, inputB: torch.Tensor) -> torch.Tensor:
        input = torch.cat((inputA, inputB), dim=1)
        out = self.discriminator_layers(input)
        return out
