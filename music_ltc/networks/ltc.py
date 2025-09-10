import torch as th
from liquid_networks.networks import AbstractLiquidRecurrent
from torch.nn import functional as th_f


class SimpleLTC(AbstractLiquidRecurrent[th.Tensor]):
    def __init__(
        self,
        neuron_number: int,
        unfolding_steps: int,
        delta_t: float,
        input_size: int,
    ) -> None:
        super().__init__(neuron_number, input_size, unfolding_steps, th_f.tanh, delta_t)

    def _process_input(self, i: th.Tensor) -> th.Tensor:
        return i

    def _output_processing(self, out: th.Tensor) -> th.Tensor:
        return out

    def _sequence_processing(self, outputs: list[th.Tensor]) -> th.Tensor:
        # (B, C, T)
        return th.stack(outputs, dim=-1)
