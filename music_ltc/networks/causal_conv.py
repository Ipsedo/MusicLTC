import torch as th
import torch.nn.functional as th_f
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
        )
        self.__out_channels = out_channels

    # pylint: disable=arguments-renamed
    def forward(self, x: th.Tensor) -> th.Tensor:
        return super().forward(
            th_f.pad(x, ((self.kernel_size[0] - 1) * self.dilation[0], 0))
        )

    def get_out_channels(self) -> int:
        return self.__out_channels


class CausalConvStrideBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__(
            weight_norm(
                CausalConv1d(in_channels, out_channels, kernel_size, 1, 1)
            ),
            nn.Mish(),
            weight_norm(
                CausalConv1d(out_channels, out_channels, kernel_size, 2, 1)
            ),
            nn.Mish(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


# Transposed Conv


class CausalConvTranspose1d(nn.ConvTranspose1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int = 1,
        output_padding: int = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            output_padding=output_padding,
        )
        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels

    # pylint: disable=arguments-renamed
    def forward(
        self, x: th.Tensor, output_size: list[int] | None = None
    ) -> th.Tensor:
        out = super().forward(x)
        cut = (self.kernel_size[0] - 1) * self.dilation[0]
        out = out[:, :, :-cut]
        return out


class CausalConvTransposeStrideBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3
    ) -> None:
        super().__init__(
            weight_norm(
                CausalConvTranspose1d(
                    in_channels, in_channels, kernel_size, 2, 1, 1
                )
            ),
            nn.Mish(),
            weight_norm(
                CausalConvTranspose1d(
                    in_channels, out_channels, kernel_size, 1, 1, 0
                )
            ),
            nn.Mish(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels
