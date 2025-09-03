import torch as th
from liquid_networks.networks import AbstractLiquidRecurrent
from torch.nn import functional as th_f

from .causal_conv import CausalConvStrideBlock, CausalConvTransposeStrideBlock
from .conv import Conv1dBlock, Conv1dOutputBlock
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
            time_size, Conv1dBlock(channels, hidden_channels[0][0], 1)
        )
        self.__encoder = SequentialTimeWrapper(
            time_size,
            [
                CausalConvStrideBlock(c_i, c_o, 4)
                for c_i, c_o in hidden_channels
            ],
        )

        self.__to_decoder = TimeWrapper(
            time_size,
            Conv1dBlock(neuron_number, hidden_channels[-1][1], 1),
        )

        self.__decoder = SequentialTimeWrapper(
            time_size,
            [
                CausalConvTransposeStrideBlock(c_i, c_o, 4)
                for c_o, c_i in reversed(hidden_channels)
            ],
        )
        self.__last_layer = Conv1dOutputBlock(
            hidden_channels[0][0], channels * 2, 1
        )

        self.__time_emb: th.Tensor = th.empty(1)

    # pylint: disable=arguments-renamed
    def _process_input(
        self, input_and_diffusion_step: tuple[th.Tensor, th.Tensor]
    ) -> th.Tensor:
        input_audio, t = input_and_diffusion_step

        transposed_input = input_audio.transpose(1, 2)

        self.__time_emb = self.__embedding(t)

        encoded_input: th.Tensor = self.__first_layer(
            transposed_input, self.__time_emb
        )
        encoded_input = self.__encoder(encoded_input, self.__time_emb)

        return encoded_input.transpose(1, 2)

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        stacked_outputs = th.stack(outputs, dim=-1)
        decoded_outputs: th.Tensor = self.__to_decoder(
            stacked_outputs, self.__time_emb
        )
        decoded_outputs = self.__decoder(decoded_outputs, self.__time_emb)
        decoded_outputs = self.__last_layer(decoded_outputs)
        return decoded_outputs.transpose(1, 2)
