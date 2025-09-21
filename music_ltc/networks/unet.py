from functools import partial

import torch as th
from torch import nn

from .conv import (
    Conv1dBlock,
    ConvStrideBlock,
    ConvTranspose1dBlock,
    ConvTransposeStrideBlock,
    OutputConv1dBlock,
)
from .ltc import SimpleLTC
from .time import FiLM, SequentialTimeWrapper, SinusoidTimeEmbedding, TimeWrapper
from .weight_init import weights_init


class WaveUNetLTC(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        delta_t: float,
        channels: list[tuple[int, int]],
        nb_diffusion_steps: int,
        time_size: int,
    ) -> None:
        super().__init__()

        self.__embedding = SinusoidTimeEmbedding(nb_diffusion_steps, time_size)

        # encoder
        self.__encoder = nn.ModuleList(
            [
                SequentialTimeWrapper(
                    time_size,
                    [Conv1dBlock(c_i, c_o), Conv1dBlock(c_o, c_o)],
                )
                for c_i, c_o in channels
            ]
        )

        self.__encoder_down = nn.ModuleList(
            [TimeWrapper(time_size, ConvStrideBlock(c_o, c_o)) for _, c_o in channels]
        )

        # LTC
        self.__ltc = SimpleLTC(neuron_number, unfolding_steps, delta_t, channels[-1][1])
        self.__ltc_film = FiLM(time_size, neuron_number)

        # decoder
        self.__to_decoder = TimeWrapper(
            time_size,
            Conv1dBlock(neuron_number, channels[-1][1]),
        )

        decoder_channels = channels.copy()
        decoder_channels[0] = (channels[0][1], channels[0][1])

        self.__decoder_up = nn.ModuleList(
            [
                TimeWrapper(time_size, ConvTransposeStrideBlock(c_i, c_i))
                for _, c_i in reversed(decoder_channels)
            ]
        )

        self.__decoder = nn.ModuleList(
            [
                SequentialTimeWrapper(
                    time_size, [ConvTranspose1dBlock(c_i * 2, c_i), ConvTranspose1dBlock(c_i, c_o)]
                )
                for c_o, c_i in reversed(decoder_channels)
            ]
        )

        self.__to_eps = OutputConv1dBlock(decoder_channels[0][0], channels[0][0])
        self.__to_v = nn.Sequential(
            OutputConv1dBlock(decoder_channels[0][0], channels[0][0]), nn.Sigmoid()
        )

        # init
        self.apply(partial(weights_init, tau_0=delta_t))

    def forward(self, x_t: th.Tensor, t: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        out = x_t.transpose(1, 2)

        time_emb = self.__embedding(t)

        bypasses = []

        for block, down in zip(self.__encoder, self.__encoder_down):
            out = block(out, time_emb)
            bypasses.append(out)
            out = down(out, time_emb)

        # (B, T, C)
        out = out.transpose(1, 2)
        out = self.__ltc(out)
        out = self.__ltc_film(out, time_emb)
        out = self.__to_decoder(out, time_emb)

        for block, up, bypass in zip(self.__decoder, self.__decoder_up, reversed(bypasses)):
            out = up(out, time_emb)
            out = th.cat([out, bypass], dim=1)
            out = block(out, time_emb)

        eps_theta = self.__to_eps(out)
        v_theta = self.__to_v(out)

        return th.transpose(eps_theta, 1, 2), th.transpose(v_theta, 1, 2)
