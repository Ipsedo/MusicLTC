import torch as th


def var_kl_div(
    p_mu: th.Tensor,
    p_var: th.Tensor,
    q_mu: th.Tensor,
    q_var: th.Tensor,
    epsilon: float = 1e-12,
) -> th.Tensor:
    return 0.5 * (
        (q_var + (q_mu - p_mu) ** 2) / (p_var + epsilon)
        + (th.log(p_var + epsilon) - th.log(q_var + epsilon))
        - 1.0
    )


def mse(p: th.Tensor, q: th.Tensor) -> th.Tensor:
    return th.pow(p - q, 2.0)
