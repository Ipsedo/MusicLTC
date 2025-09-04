import pytest
import torch as th

from music_ltc.networks.denoiser import Denoiser


# pylint: disable=duplicate-code
@pytest.mark.parametrize("steps", [4, 6])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("length", [16, 32])
@pytest.mark.parametrize("time_size", [2, 4])
def test_denoiser(
    steps: int,
    batch_size: int,
    length: int,
    time_size: int,
) -> None:
    in_channels = 2
    denoiser = Denoiser(steps, time_size, in_channels, [(4, 4)], 4, 6, 1.0)

    denoiser.eval()

    device = "cpu"

    x_t = th.randn(
        batch_size,
        length,
        in_channels,
        device=device,
    )
    t = th.randint(
        0,
        steps,
        (batch_size,),
        device=device,
    )

    eps, v = denoiser(x_t, t)

    assert len(eps.size()) == 3
    assert eps.size() == (batch_size, length, in_channels)

    assert len(v.size()) == 3
    assert v.size() == (batch_size, length, in_channels)

    prior_mu, prior_var = denoiser.prior(x_t, t, eps, v)

    assert len(prior_mu.size()) == 3
    assert prior_mu.size() == (batch_size, length, in_channels)

    assert len(prior_var.size()) == 3
    assert prior_var.size() == (batch_size, length, in_channels)
    assert th.all(th.gt(prior_var, 0.0))

    x_t = th.randn(
        batch_size,
        length,
        in_channels,
        device=device,
    )

    x_0 = denoiser.sample(x_t)

    assert len(x_0.size()) == 3
    assert x_0.size() == (batch_size, length, in_channels)

    x_0 = denoiser.fast_sample(x_t, steps // 2)

    assert len(x_0.size()) == 3
    assert x_0.size() == (batch_size, length, in_channels)
