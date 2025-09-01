import torch as th
from liquid_networks.networks import AbstractLiquidRecurrent
from torch import nn
from torch.nn import functional as th_f

from .conv import (
    CausalConvBlock,
    CausalConvStrideBlock,
    CausalConvTranspose1d,
    CausalConvTransposeBlock,
)
from .time import SequentialTimeWrapper, SinusoidTimeEmbedding, TimeWrapper


class WaveLTC(AbstractLiquidRecurrent[tuple[th.Tensor, th.Tensor]]):
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
        super().__init__(
            neuron_number,
            hidden_channels[-1][1],
            unfolding_steps,
            th_f.mish,
            delta_t,
        )

        self.__embedding = SinusoidTimeEmbedding(nb_diffusion_steps, time_size)

        self.__first_layer = TimeWrapper(
            time_size, CausalConvBlock(channels, hidden_channels[0][0])
        )
        self.__encoder = SequentialTimeWrapper(
            time_size,
            [CausalConvStrideBlock(c_i, c_o) for c_i, c_o in hidden_channels],
        )

        self.__to_decoder = nn.Sequential(
            CausalConvTranspose1d(neuron_number, hidden_channels[-1][1], 3, 1),
            nn.Mish(),
            nn.InstanceNorm1d(hidden_channels[-1][1]),
        )
        self.__decoder = nn.Sequential(
            *[
                CausalConvTransposeBlock(c_i, c_o)
                for c_o, c_i in reversed(hidden_channels)
            ]
        )
        self.__last_layer = CausalConvTranspose1d(
            hidden_channels[0][0], channels * 2, 3, 1
        )

        self.__to_first_token = nn.Sequential(
            nn.Linear(time_size, hidden_channels[-1][1]),
            nn.Mish(),
            nn.LayerNorm(hidden_channels[-1][1]),
        )

    # pylint: disable=arguments-renamed
    def _process_input(
        self, input_and_diffusion_step: tuple[th.Tensor, th.Tensor]
    ) -> th.Tensor:
        i, t = input_and_diffusion_step

        time_emb = self.__embedding(t)

        encoded_input = self.__first_layer(i.transpose(1, 2), time_emb)
        encoded_input = self.__encoder(encoded_input, time_emb).transpose(1, 2)

        first_token = self.__to_first_token(time_emb).unsqueeze(1)

        return th.cat([first_token, encoded_input], dim=1)

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        stacked_outputs = th.stack(outputs[1:], dim=-1)
        decoded_outputs: th.Tensor = self.__to_decoder(stacked_outputs)
        decoded_outputs = self.__decoder(decoded_outputs)
        decoded_outputs = self.__last_layer(decoded_outputs)
        return decoded_outputs.transpose(1, 2)
