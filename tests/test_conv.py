import pytest
import torch as th

from music_ltc.networks.causal_conv import (
    CausalConv1d,
    CausalConvStrideBlock,
    CausalConvTranspose1d,
    CausalConvTransposeStrideBlock,
)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [10, 20])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_causal_conv_1d(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv = CausalConv1d(in_channels, out_channels, 3, 1, 1)

    x = th.randn(batch_size, in_channels, length)

    out = conv(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [10, 20])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_causal_conv_block(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv = CausalConvStrideBlock(in_channels, out_channels)

    x = th.randn(batch_size, in_channels, length)

    out = conv(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length // 2)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [10, 20])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_causal_conv_tr_1d(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv_tr = CausalConvTranspose1d(in_channels, out_channels, 3, 1, 1)

    x = th.randn(batch_size, in_channels, length)

    out = conv_tr(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length)


@pytest.mark.parametrize("in_channels", [1, 2])
@pytest.mark.parametrize("out_channels", [1, 2])
@pytest.mark.parametrize("length", [10, 20])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_causal_conv_tr_block(
    in_channels: int, out_channels: int, length: int, batch_size: int
) -> None:
    conv_tr = CausalConvTransposeStrideBlock(in_channels, out_channels)

    x = th.randn(batch_size, in_channels, length)

    out = conv_tr(x)

    assert len(out.size()) == 3
    assert out.size() == (batch_size, out_channels, length * 2)
