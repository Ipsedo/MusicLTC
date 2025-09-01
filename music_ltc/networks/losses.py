import torch as th


def normal_kl_div(
    mu_1: th.Tensor,
    var_1: th.Tensor,
    mu_2: th.Tensor,
    var_2: th.Tensor,
    epsilon: float = 1e-12,
) -> th.Tensor:
    return (
        th.log(var_2 + epsilon) / 2.0
        - th.log(var_1 + epsilon) / 2.0
        + (var_1 + th.pow(mu_1 - mu_2, 2.0)) / (2 * var_2 + epsilon)
        - 0.5
    )


def mse(p: th.Tensor, q: th.Tensor) -> th.Tensor:
    return th.pow(p - q, 2.0)
