from os import mkdir
from os.path import exists, isdir, join

import mlflow
import torch as th
import torchaudio as th_audio
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.dataset import AudioDataset
from .networks.losses import mse, normal_kl_div
from .options import ModelOptions, TrainOptions


def train_model(
    model_options: ModelOptions, train_options: TrainOptions
) -> None:
    mlflow.set_experiment("music_diffusion_ltc")

    with mlflow.start_run(run_name="train_debug"):

        if not exists(train_options.output_dir):
            mkdir(train_options.output_dir)
        elif not isdir(train_options.output_dir):
            raise NotADirectoryError(
                f'"{train_options.output_dir}" is not a directory.'
            )

        mlflow.log_params(
            {
                **dict(model_options),
                **dict(train_options),
            }
        )

        if train_options.cuda:
            th.backends.cudnn.benchmark = True

        noiser = model_options.get_noiser()
        denoiser = model_options.get_denoiser()

        print(f"Parameters count = {denoiser.count_parameters()}")

        optim = th.optim.Adam(
            denoiser.parameters(),
            lr=train_options.learning_rate,
        )

        if train_options.cuda:
            noiser.cuda()
            denoiser.cuda()

        dataset = AudioDataset(train_options.dataset_path)

        dataloader = DataLoader(
            dataset,
            batch_size=train_options.batch_size,
            shuffle=True,
            num_workers=train_options.dataloader_workers,
            drop_last=True,
            pin_memory=True,
        )

        device = th.device("cuda" if train_options.cuda else "cpu")

        nb_batches = len(dataset) // train_options.batch_size

        for e in range(train_options.epochs):

            tqdm_bar = tqdm(dataloader)

            for i, x_0 in enumerate(tqdm_bar):
                x_0 = x_0.to(device)

                t = th.randint(
                    0,
                    model_options.diffusion_steps,
                    (train_options.batch_size,),
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

                loss = th.mean(loss_kl + loss_mse)

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                grad_norm = denoiser.grad_norm()

                tqdm_bar.set_description(
                    f"Epoch {e} / {train_options.epochs - 1} - "
                    f"loss = {loss.item():.6f}, "
                    f"mse = {loss_mse.mean().item():.6f}, "
                    f"kl = {loss_kl.mean().item():.6f}, "
                    f"grad_norm = {grad_norm:.6f}"
                )

                mlflow.log_metrics(
                    {
                        "loss": loss.item(),
                        "mse": loss_mse.mean().item(),
                        "kl": loss_kl.mean().item(),
                        "grad_norm": grad_norm,
                    },
                    step=e * nb_batches + i,
                )

            # save models
            th.save(
                denoiser.state_dict(),
                join(train_options.output_dir, f"denoiser_{e}.pth"),
            )
            th.save(
                noiser.state_dict(),
                join(train_options.output_dir, f"noiser_{e}.pth"),
            )
            th.save(
                optim.state_dict(),
                join(train_options.output_dir, f"optim_{e}.pth"),
            )

            # generate audio
            x_t = th.randn(
                train_options.nb_audios_to_generate,
                2**17,
                model_options.channels,
                device=device,
            )

            x_0 = denoiser.sample(x_t, verbose=True)

            th.save(
                x_0,
                join(train_options.output_dir, f"waveform_{e}.pt"),
            )

            for i in range(train_options.nb_audios_to_generate):
                waveform_tensor = x_0[i].detach().cpu()
                th_audio.save_with_torchcodec(
                    join(train_options.output_dir, f"audio_{e}_{i}.wav"),
                    waveform_tensor,
                    train_options.sample_rate,
                    channels_first=False,
                )
