import torch
import torch.nn as nn

from .blocks import DownsampleBlock, UpsampleBlock

class Generator(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int, int],
        in_channels: list[int],
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels

        down_blocks = []
        down_in_channels = in_channels
        for idx in range(len(down_in_channels)):
            down_blocks.append(
                DownsampleBlock(
                    in_channels=self.img_size[0] if idx == 0 else down_in_channels[idx - 1],
                    out_channels=down_in_channels[idx],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    use_instance_norm=idx != 0,
                    dropout_p=0.25,
                )
            )
        self.down_blocks = nn.ModuleList(down_blocks)

        self.mid_blocks = DownsampleBlock(
            in_channels=down_in_channels[-1],
            out_channels=down_in_channels[-1],
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
            use_instance_norm=False,
            dropout_p=0.5,
        )

        up_blocks = []
        up_in_channels = list(reversed(self.in_channels))
        for idx in range(len(up_in_channels)):
            up_blocks.append(
                UpsampleBlock(
                    in_channels=up_in_channels[idx] if idx == 0 else up_in_channels[idx - 1] * 2,
                    out_channels=up_in_channels[idx],
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                    use_instance_norm=True,
                    dropout_p=0.25,
                )
            )
        self.up_blocks = nn.ModuleList(up_blocks)

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(self.in_channels[0] * 2, self.img_size[0], 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        skips = []
        out = input
        for down_block in self.down_blocks:
            out = down_block(out)
            skips.append(out)
        out = self.mid_blocks(out)
        for idx, up_block in enumerate(self.up_blocks):
            out = up_block(out)
            out = torch.cat((out, skips[-(idx + 1)]), dim=1)
        return self.conv(out)