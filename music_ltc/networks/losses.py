import torch as th


def var_kl_div(
    var_pred: th.Tensor,
    var_target: th.Tensor,
    epsilon: float = 1e-12,
) -> th.Tensor:
    return 0.5 * ((var_target / (var_pred + epsilon)) - 1 - th.log(var_target / (var_pred + epsilon)))


def mse(p: th.Tensor, q: th.Tensor) -> th.Tensor:
    return th.pow(p - q, 2.0)
