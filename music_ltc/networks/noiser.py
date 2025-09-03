import torch as th

from .diffusion import Diffuser, select_time_scheduler


class Noiser(Diffuser):
    def forward(
        self, x_0: th.Tensor, t: th.Tensor, eps: th.Tensor | None = None
    ) -> tuple[th.Tensor, th.Tensor]:
        assert len(x_0.size()) == 3
        assert len(t.size()) == 1
        assert x_0.size(0) == t.size(0)

        if eps is None:
            eps = th.rand_like(x_0, device=next(self.buffers()).device)

        sqrt_alphas_cum_prod = select_time_scheduler(
            self._sqrt_alphas_cum_prod, t
        )
        sqrt_one_minus_alphas_cum_prod = select_time_scheduler(
            self._sqrt_one_minus_alphas_cum_prod, t
        )

        x_t = sqrt_alphas_cum_prod * x_0 + eps * sqrt_one_minus_alphas_cum_prod

        return x_t, eps

    def __mu(self, x_t: th.Tensor, x_0: th.Tensor, t: th.Tensor) -> th.Tensor:
        return super()._mu_tiddle(x_t, x_0, t)

    def __var(self, t: th.Tensor) -> th.Tensor:

        betas: th.Tensor = select_time_scheduler(self._betas_tiddle, t)

        return betas

    def posterior(
        self,
        x_t: th.Tensor,
        x_0: th.Tensor,
        t: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 3
        assert len(x_0.size()) == 3
        assert len(t.size()) == 1

        assert x_t.size(0) == t.size(0)
        assert x_t.size() == x_0.size()

        return self.__mu(x_t, x_0, t), self.__var(t)
