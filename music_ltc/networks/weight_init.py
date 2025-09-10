from liquid_networks.networks.liquid_cell import CellModel, LiquidCell
from torch import nn

from .time import FiLM


def __init_film_linear(lin_model: nn.Module) -> None:
    if isinstance(lin_model, nn.Linear):
        nn.init.zeros_(lin_model.weight)
        if lin_model.bias is not None:
            nn.init.zeros_(lin_model.bias)


def weights_init(m: nn.Module, ltc_unfolding_steps: int) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, FiLM):
        m.apply(__init_film_linear)
    elif isinstance(m, CellModel):
        nn.init.normal_(m.weight, 0.0, 1.0 / ltc_unfolding_steps)

        m.recurrent_weight.data.normal_(0.0, 1.0 / ltc_unfolding_steps)
        m.recurrent_weight.data.abs_()

        nn.init.normal_(m.bias, 0.0, 1e-1)
    elif isinstance(m, LiquidCell):
        nn.init.normal_(m.a, 0.0, 1.0 / ltc_unfolding_steps)
        nn.init.normal_(m.log_tau, 0.0, 1.0 / ltc_unfolding_steps)
