from statistics import mean

import numpy as np
import torch as th
from tqdm import tqdm

from .diffusion import Diffuser, select_time_scheduler
from .ltc import WaveLTC


class Denoiser(Diffuser):
    def __init__(
        self,
        steps: int,
        time_size: int,
        channels: int,
        hidden_channels: list[tuple[int, int]],
        neuron_number: int,
        unfolding_steps: int,
        delta_t: float,
    ) -> None:
        super().__init__(steps)

        self.__channels = channels

        self._sqrt_alpha: th.Tensor
        self._sqrt_betas: th.Tensor

        self.register_buffer(
            "_sqrt_alpha",
            th.sqrt(self._alphas),
        )

        self.register_buffer(
            "_sqrt_betas",
            th.sqrt(self._betas),
        )

        self.__network = WaveLTC(
            neuron_number,
            unfolding_steps,
            delta_t,
            channels,
            hidden_channels,
            steps,
            time_size,
        )

    def forward(
        self, x_t: th.Tensor, t: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 3
        assert len(t.size()) == 1
        assert x_t.size(0) == t.size(0)

        eps_and_v_theta = self.__network((x_t, t))
        eps_theta, v_theta = th.chunk(eps_and_v_theta, 2, dim=-1)

        return eps_theta, v_theta

    def __x0_from_noise(
        self,
        x_t: th.Tensor,
        eps: th.Tensor,
        t: th.Tensor,
        alphas_cum_prod: th.Tensor | None,
    ) -> th.Tensor:
        alphas_cum_prod = (
            select_time_scheduler(self._alphas_cum_prod, t)
            if alphas_cum_prod is None
            else alphas_cum_prod
        )
        x_0: th.Tensor = (x_t - eps * th.sqrt(1 - alphas_cum_prod)) / th.sqrt(
            alphas_cum_prod
        )
        return th.clip(x_0, -1.0, 1.0)

    def __mu_clipped(
        self,
        x_t: th.Tensor,
        eps_theta: th.Tensor,
        t: th.Tensor,
        alphas: th.Tensor | None = None,
        betas: th.Tensor | None = None,
        alphas_cum_prod: th.Tensor | None = None,
        alphas_cum_prod_prev: th.Tensor | None = None,
    ) -> th.Tensor:
        x_0_clipped = self.__x0_from_noise(x_t, eps_theta, t, alphas_cum_prod)

        mu = self._mu_tiddle(
            x_t,
            x_0_clipped,
            t,
            alphas,
            betas,
            alphas_cum_prod,
            alphas_cum_prod_prev,
        )

        return mu

    def __mu(
        self, x_t: th.Tensor, eps_theta: th.Tensor, t: th.Tensor
    ) -> th.Tensor:

        mu: th.Tensor = (
            x_t
            - eps_theta
            * select_time_scheduler(self._betas, t)
            / select_time_scheduler(self._sqrt_one_minus_alphas_cum_prod, t)
        ) / select_time_scheduler(self._sqrt_alpha, t)
        return mu

    def __var(
        self,
        v: th.Tensor,
        t: th.Tensor,
        betas: th.Tensor | None = None,
        betas_tiddle: th.Tensor | None = None,
    ) -> th.Tensor:

        betas = (
            select_time_scheduler(self._betas, t) if betas is None else betas
        )
        betas_tiddle = (
            select_time_scheduler(self._betas_tiddle, t)
            if betas_tiddle is None
            else betas_tiddle
        )

        return th.exp(v * th.log(betas) + (1.0 - v) * th.log(betas_tiddle))

    def prior(
        self,
        x_t: th.Tensor,
        t: th.Tensor,
        eps_theta: th.Tensor,
        v_theta: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        assert len(x_t.size()) == 3
        assert len(t.size()) == 1
        assert len(eps_theta.size()) == 3
        assert len(v_theta.size()) == 3

        assert x_t.size(2) == self.__channels
        assert x_t.size() == eps_theta.size()
        assert x_t.size() == v_theta.size()

        return self.__mu(x_t, eps_theta, t), self.__var(v_theta, t)

    @th.no_grad()
    def sample(self, x_t: th.Tensor, verbose: bool = False) -> th.Tensor:
        assert len(x_t.size()) == 3
        assert x_t.size(2) == self.__channels

        device = next(self.parameters()).device

        times = list(reversed(range(self._steps)))
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for t in tqdm_bar:
            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            t_tensor = th.tensor([t] * x_t.size(0), device=device)

            eps, v = self.forward(x_t, t_tensor)

            # original sampling method
            # see : https://github.com/hojonathanho/diffusion/issues/5
            # see : https://github.com/openai/improved-diffusion/issues/64
            mu = self.__mu_clipped(x_t, eps, t_tensor)
            sigma = self.__var(v, t_tensor).sqrt()

            x_t = mu + sigma * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    @th.no_grad()
    def fast_sample(
        self, x_t: th.Tensor, n_steps: int, verbose: bool = False
    ) -> th.Tensor:
        assert len(x_t.size()) == 3
        assert x_t.size(2) == self.__channels

        device = next(self.parameters()).device

        steps = th.linspace(
            0, self._steps - 1, steps=n_steps, dtype=th.long, device=device
        )

        alphas_cum_prod_s = self._alphas_cum_prod[steps]
        alphas_cum_prod_prev_s = th.cat(
            [th.tensor([1], device=device), alphas_cum_prod_s[:-1]], dim=0
        )

        betas_s = 1.0 - alphas_cum_prod_s / alphas_cum_prod_prev_s
        betas_s = th.clamp_max(betas_s, 0.999)

        betas_tiddle_s = (
            betas_s
            * (1.0 - alphas_cum_prod_prev_s)
            / (1.0 - alphas_cum_prod_s)
        )
        betas_tiddle_s = th.clamp_min(betas_tiddle_s, betas_tiddle_s[1])

        alphas_s = 1.0 - betas_s

        times: list[int] = steps.flip(0).cpu().numpy().tolist()
        tqdm_bar = tqdm(times, disable=not verbose, leave=False)

        for s_t, t in enumerate(tqdm_bar):
            s_t = len(times) - s_t - 1

            z = (
                th.randn_like(x_t, device=device)
                if t > 0
                else th.zeros_like(x_t, device=device)
            )

            t_tensor = th.tensor([t] * x_t.size(0), device=device)

            eps, v = self.forward(x_t, t_tensor)

            mu = self.__mu_clipped(
                x_t,
                eps,
                t_tensor,
                alphas_s[s_t],
                betas_s[s_t],
                alphas_cum_prod_s[s_t],
                alphas_cum_prod_prev_s[s_t],
            )

            var = self.__var(v, t_tensor, betas_s[s_t], betas_tiddle_s[s_t])

            x_t = mu + var.sqrt() * z

            tqdm_bar.set_description(
                f"Generate {x_t.size(0)} data with size {tuple(x_t.size()[1:])}"
            )

        return x_t

    def count_parameters(self) -> int:
        return int(
            sum(
                np.prod(p.size()) for p in self.parameters() if p.requires_grad
            )
        )

    def grad_norm(self) -> float:
        return float(
            mean(
                p.grad.norm().item()
                for p in self.parameters()
                if p.grad is not None
            )
        )
