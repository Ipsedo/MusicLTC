from os.path import join

import mlflow
import torch as th
import torchaudio as th_audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import AudioDataset
from .networks.denoiser import Denoiser
from .networks.losses import mse, normal_kl_div
from .networks.noiser import Noiser


def train() -> None:
    mlflow.set_experiment("music_diffusion_ltc")

    cuda = True
    dataset_path = (
        "/run/media/samuel/M2_nvme_gen4/music_diffusion/bach_waveform_16000Hz"
    )
    output_dir = "/home/samuel/PycharmProjects/MusicLTC/outputs/train_bach"
    steps = 1024
    batch_size = 32
    epochs = 100
    nb_samples = 3
    in_channels = 2
    sample_rate = 16000

    with mlflow.start_run(run_name="train_debug"):

        if cuda:
            th.backends.cudnn.benchmark = True

        noiser = Noiser(steps)
        denoiser = Denoiser(
            steps,
            16,
            in_channels,
            [
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 64),
                (64, 96),
                (96, 128),
                (128, 128),
            ],
            64,
            6,
            1.0,
        )

        print(f"Parameters count = {denoiser.count_parameters()}")

        optim = th.optim.Adam(
            denoiser.parameters(),
            lr=1e-4,
        )

        if cuda:
            noiser.cuda()
            denoiser.cuda()

        dataset = AudioDataset(dataset_path)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=6,
            drop_last=True,
            pin_memory=True,
        )

        device = "cuda" if cuda else "cpu"

        for e in range(epochs):

            tqdm_bar = tqdm(dataloader)

            for x_0 in tqdm_bar:

                if cuda:
                    x_0 = x_0.cuda()

                t = th.randint(
                    0,
                    steps,
                    (batch_size,),
                    device=device,
                )

                x_t, eps = noiser(x_0, t)
                eps_theta, v_theta = denoiser(x_t, t)

                loss_mse = mse(eps, eps_theta)

                q_mu, q_var = noiser.posterior(x_t, x_0, t)
                p_mu, p_var = denoiser.prior(
                    x_t, t, eps_theta.detach(), v_theta
                )

                loss_kl = normal_kl_div(q_mu, q_var, p_mu, p_var)
                # loss_nll = negative_log_likelihood(x_0, p_mu, p_var)
                # loss_nll = discretized_nll(x_0.unsqueeze(1), p_mu, p_var)
                # loss_vlb = th.where(th.eq(t, 0), loss_nll, loss_kl)

                loss = loss_kl + loss_mse
                loss = loss.mean()

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                grad_norm = denoiser.grad_norm()

                tqdm_bar.set_description(
                    f"Epoch {e} / {epochs - 1} - "
                    f"loss = {loss.mean().item():.6f}, "
                    f"mse = {loss_mse.mean().item():.6f}, "
                    f"kl = {loss_kl.mean().item():.6f}, "
                    f"grad_norm = {grad_norm:.6f}"
                )

            # save and generate
            x_t = th.randn(
                nb_samples,
                2**17,
                in_channels,
                device=device,
            )

            with th.no_grad():
                x_0 = denoiser.sample(x_t, verbose=True)

            th.save(
                x_0,
                join(output_dir, f"waveform_{e}.pt"),
            )

            for i in range(nb_samples):
                waveform_tensor = x_0[i].detach().cpu()
                th_audio.save(
                    join(output_dir, f"audio_{e}_{i}.wav"),
                    waveform_tensor,
                    sample_rate,
                    channels_first=False,
                )
