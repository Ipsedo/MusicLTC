import pytest
import torch as th

from music_ltc.networks.noiser import Noiser

# pylint: disable=duplicate-code


@pytest.mark.parametrize("steps", [2, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("length", [16, 32])
def test_noiser(
    steps: int,
    batch_size: int,
    channels: int,
    length: int,
) -> None:
    noiser = Noiser(steps)

    device = "cpu"

    x_0 = th.randn(
        batch_size,
        channels,
        length,
        device=device,
    )
    t = th.randint(
        0,
        steps,
        (batch_size,),
        device=device,
    )

    x_t, eps = noiser(x_0, t)

    assert len(x_t.size()) == 3
    assert x_t.size() == (batch_size, channels, length)

    assert len(eps.size()) == 3
    assert eps.size() == (batch_size, channels, length)

    post_mu, post_var = noiser.posterior(x_t, x_0, t)

    assert len(post_mu.size()) == 3
    assert post_mu.size() == (batch_size, channels, length)

    assert len(post_var.size()) == 3
    assert post_var.size() == (batch_size, 1, 1)
    assert th.all(th.gt(post_var, 0))
