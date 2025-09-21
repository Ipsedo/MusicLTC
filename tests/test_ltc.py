import pytest
import torch as th

from music_ltc.networks.unet import WaveUNetLTC


@pytest.mark.parametrize(
    "hidden_channels",
    [
        [
            (1, 8),
            (8, 16),
        ],
        [(2, 8)],
    ],
)
@pytest.mark.parametrize("length", [16, 32])
@pytest.mark.parametrize("batch_size", [2, 3])
def test_wave_ltc(
    hidden_channels: list[tuple[int, int]],
    length: int,
    batch_size: int,
) -> None:
    diff_step = 16
    time_size = 4

    wave_ltc = WaveUNetLTC(
        4,
        6,
        1.0,
        hidden_channels,
        diff_step,
        time_size,
    )

    x = th.randn(batch_size, length, hidden_channels[0][0])
    t = th.randint(0, diff_step, (batch_size,))

    out_eps, out_v = wave_ltc(x, t)

    assert len(out_eps.size()) == 3
    assert out_eps.size() == (batch_size, length, hidden_channels[0][0])

    assert len(out_v.size()) == 3
    assert out_v.size() == (batch_size, length, hidden_channels[0][0])
