import math

from liquid_networks.networks.liquid_cell import CellModel, LiquidCell
from torch import nn

from .time import FiLM


def __init_film_linear(lin_model: nn.Module) -> None:
    if isinstance(lin_model, nn.Linear):
        nn.init.zeros_(lin_model.weight)
        if lin_model.bias is not None:
            nn.init.zeros_(lin_model.bias)


def weights_init(m: nn.Module, tau_0: float) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, FiLM):
        m.apply(__init_film_linear)
    elif isinstance(m, CellModel):
        nn.init.xavier_normal_(m.weight, gain=1.0)
        nn.init.normal_(m.bias, 0.0, 0.25)

        mu = 1.0 / max(tau_0, 1e-6)
        sigma = 0.25 * mu
        m.recurrent_weight.data.normal_(mu, sigma).abs_()
    elif isinstance(m, LiquidCell):
        nn.init.normal_(m.a, 0.0, 0.1)
        nn.init.normal_(m.raw_tau, math.log(math.exp(tau_0) - 1.0), 0.5)
