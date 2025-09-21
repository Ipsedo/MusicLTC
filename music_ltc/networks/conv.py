import torch as th
from torch import nn


class Conv1dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, 3, 1, 1),
            nn.ELU(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


class OutputConv1dBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.__conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)

        self.__out_channels = out_channels

    def forward(self, x: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__conv(x)
        return out

    def get_out_channels(self) -> int:
        return self.__out_channels


class ConvTranspose1dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 3, 1, 1),
            nn.ELU(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


# stride


class ConvStrideBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv1d(in_channels, out_channels, 8, 4, 2),
            nn.ELU(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


# Transposed Conv


class ConvTransposeStrideBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.ConvTranspose1d(in_channels, out_channels, 8, 4, 2, 0),
            nn.ELU(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels
