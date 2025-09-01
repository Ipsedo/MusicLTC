import pytest
import torch as th

from music_ltc.networks.ltc import WaveLTC


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("length", [16, 32])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_wave_ltc(channels: int, length: int, batch_size: int) -> None:
    diff_step = 16
    time_size = 4

    wave_ltc = WaveLTC(
        4,
        6,
        1.0,
        channels,
        [
            (4, 8),
            (8, 16),
        ],
        diff_step,
        time_size,
    )

    x = th.randn(batch_size, length, channels)
    t = th.randint(0, diff_step, (batch_size,))

    out = wave_ltc((x, t))

    assert len(out.size()) == 3
    assert out.size() == (batch_size, length, channels * 2)
