import pytest
import torch as th

from music_ltc.networks.conv import ConvStrideBlock, ConvTransposeStrideBlock


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [16, 32])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_conv_block(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv = ConvStrideBlock(in_channels, out_channels)

    x = th.randn(batch_size, in_channels, length)

    out = conv(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length // 4)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [16, 32])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_conv_tr_block(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv_tr = ConvTransposeStrideBlock(in_channels, out_channels)

    x = th.randn(batch_size, in_channels, length)

    out = conv_tr(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length * 4)
