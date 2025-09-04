import torch as th
from torch import nn

from .conv import (
    ChannelOutputBlock,
    ChannelProjectionBlock,
    ConvStrideBlock,
    ConvTransposeStrideBlock,
)
from .ltc import SimpleLTC
from .time import (
    FiLM,
    SequentialTimeWrapper,
    SinusoidTimeEmbedding,
    TimeWrapper,
)


class WaveLTC(nn.Module):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        delta_t: float,
        channels: int,
        hidden_channels: list[tuple[int, int]],
        nb_diffusion_steps: int,
        time_size: int,
    ) -> None:
        super().__init__()

        self.__embedding = SinusoidTimeEmbedding(nb_diffusion_steps, time_size)

        # encoder
        self.__first_layer = TimeWrapper(
            time_size, ChannelProjectionBlock(channels, hidden_channels[0][0])
        )
        self.__encoder = SequentialTimeWrapper(
            time_size,
            [ConvStrideBlock(c_i, c_o) for c_i, c_o in hidden_channels],
        )

        # LTC
        self.__ltc = SimpleLTC(
            neuron_number, unfolding_steps, delta_t, hidden_channels[-1][1]
        )
        self.__ltc_film = FiLM(time_size, neuron_number)

        # decoder
        self.__to_decoder = TimeWrapper(
            time_size,
            ChannelProjectionBlock(neuron_number, hidden_channels[-1][1]),
        )

        self.__decoder = SequentialTimeWrapper(
            time_size,
            [
                ConvTransposeStrideBlock(c_i, c_o)
                for c_o, c_i in reversed(hidden_channels)
            ],
        )
        self.__to_eps = ChannelOutputBlock(hidden_channels[0][0], channels)
        self.__to_v = nn.Sequential(
            ChannelOutputBlock(hidden_channels[0][0], channels), nn.Sigmoid()
        )

    def forward(
        self, x_t: th.Tensor, t: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        transposed_input = x_t.transpose(1, 2)

        time_emb = self.__embedding(t)

        encoded_input: th.Tensor = self.__first_layer(
            transposed_input, time_emb
        )
        encoded_input = self.__encoder(encoded_input, time_emb)

        # (B, T, C)
        ltc_input = encoded_input.transpose(1, 2)
        ltc_output = self.__ltc(ltc_input)
        ltc_output = self.__ltc_film(ltc_output, time_emb)

        decoded_output = self.__to_decoder(ltc_output, time_emb)
        decoded_output = self.__decoder(decoded_output, time_emb)

        eps_theta = self.__to_eps(decoded_output)
        v_theta = self.__to_v(decoded_output)

        return th.transpose(eps_theta, 1, 2), th.transpose(v_theta, 1, 2)
