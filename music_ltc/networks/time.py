import math
from typing import Iterable, Protocol

import torch as th
from torch import nn


class TimeModuleProtocol(Protocol):
    def get_out_channels(self) -> int:
        pass

    def __call__(self, x: th.Tensor) -> th.Tensor:
        pass


class SinusoidTimeEmbedding(nn.Module):
    def __init__(self, steps: int, size: int) -> None:
        super().__init__()

        pos_emb = th.zeros(steps, size)
        position = th.arange(0, steps).unsqueeze(1)
        div_term = th.exp(
            th.arange(0, size, 2, dtype=th.float)
            * th.tensor(-math.log(10000.0) / size)
        )
        pos_emb[:, 0::2] = th.sin(position.float() * div_term)
        pos_emb[:, 1::2] = th.cos(position.float() * div_term)

        self._pos_emb: th.Tensor

        self.register_buffer("_pos_emb", pos_emb)

    def forward(self, t_index: th.Tensor) -> th.Tensor:
        return th.index_select(self._pos_emb, dim=0, index=t_index)


class FiLM(nn.Module):
    def __init__(self, time_size: int, channels: int) -> None:
        super().__init__()

        self.__to_channels = nn.Sequential(
            nn.Linear(time_size, channels * 2),
            nn.Mish(),
            nn.GroupNorm(4, channels * 2),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        proj_time_emb = self.__to_channels(time_emb)
        proj_time_emb = proj_time_emb[:, :, None]
        scale, shift = th.chunk(proj_time_emb, chunks=2, dim=1)

        out = x * (scale + 1.0) + shift

        return out


class TimeWrapper(nn.Module):
    def __init__(
        self,
        time_size: int,
        module: TimeModuleProtocol,
    ) -> None:
        super().__init__()

        self.__module = module
        self.__film = FiLM(time_size, module.get_out_channels())

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        out: th.Tensor = self.__module(x)
        out = self.__film(out, time_emb)

        return out


class SequentialTimeWrapper(nn.ModuleList):
    def __init__(
        self,
        time_size: int,
        conv_layers: Iterable[TimeModuleProtocol],
    ):
        super().__init__(TimeWrapper(time_size, c) for c in conv_layers)

    def forward(self, x: th.Tensor, time_emb: th.Tensor) -> th.Tensor:
        out = x
        for m in self:
            out = m(out, time_emb)
        return out
