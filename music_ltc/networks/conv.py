from torch import nn
from torch.nn.utils.parametrizations import weight_norm


class Conv1dBlock(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)),
            nn.Mish(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


class OutputConv1dBlock(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding)

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


# stride


class ConvStrideBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            weight_norm(nn.Conv1d(in_channels, out_channels, 8, 4, 2)),
            nn.Mish(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels


# Transposed Conv


class ConvTransposeStrideBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            weight_norm(nn.ConvTranspose1d(in_channels, out_channels, 8, 4, 2, 0)),
            nn.Mish(),
        )

        self.__out_channels = out_channels

    def get_out_channels(self) -> int:
        return self.__out_channels
