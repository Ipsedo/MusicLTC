from abc import ABC, abstractmethod
from os import makedirs
from os.path import exists, isdir, join

import torch as th
import torchaudio as th_audio
from torch import nn

from .networks.denoiser import Denoiser


class AbstractSaver(ABC):
    @abstractmethod
    def save(self, output_folder: str) -> None:
        pass


class TorchSaver(AbstractSaver):
    def __init__(self, name: str, module_or_optim: nn.Module | th.optim.Optimizer) -> None:
        super().__init__()

        self.__module_or_optim = module_or_optim
        self.__name = name

    def save(self, output_folder: str) -> None:
        th.save(self.__module_or_optim.state_dict(), join(output_folder, f"{self.__name}.pth"))


class AudioSaver(AbstractSaver):
    def __init__(
        self,
        denoiser: Denoiser,
        sample_rate: int,
        nb_audio_to_generate: int,
        nb_ticks: int,
        channels: int,
        fast_sample_steps: int,
    ) -> None:
        super().__init__()

        self.__denoiser = denoiser

        self.__nb_audio_to_generate = nb_audio_to_generate
        self.__nb_ticks = nb_ticks
        self.__channels = channels
        self.__fast_sample_steps = fast_sample_steps

        self.__sample_rate = sample_rate

    def save(self, output_folder: str) -> None:
        device = next(self.__denoiser.parameters()).device

        x_t = th.randn(self.__nb_audio_to_generate, self.__nb_ticks, self.__channels, device=device)

        x_0 = self.__denoiser.fast_sample(x_t, self.__fast_sample_steps, verbose=True)

        th.save(
            x_0,
            join(output_folder, "waveform.pt"),
        )

        for i in range(self.__nb_audio_to_generate):
            waveform_tensor = x_0[i].detach().cpu()
            th_audio.save_with_torchcodec(
                join(output_folder, f"audio_{i}.wav"),
                waveform_tensor,
                self.__sample_rate,
                channels_first=False,
            )


class SaveManager:
    def __init__(
        self,
        savers: list[AbstractSaver],
        output_folder_path: str,
        save_every: int,
    ) -> None:
        self.__savers = savers

        self.__save_every = save_every

        self.__output_folder_path = output_folder_path

        self.__curr_save = 0
        self.__curr_iteration = 0

    def saving_attempt(self) -> None:
        if self.__curr_iteration % self.__save_every == self.__save_every - 1:
            if not exists(self.__output_folder_path):
                makedirs(self.__output_folder_path)
            elif not isdir(self.__output_folder_path):
                raise NotADirectoryError(self.__output_folder_path)

            curr_save_folder = join(self.__output_folder_path, f"save_{self.__curr_save}")

            if not exists(curr_save_folder):
                makedirs(curr_save_folder)
            elif not isdir(curr_save_folder):
                raise NotADirectoryError(curr_save_folder)

            for saver in self.__savers:
                saver.save(curr_save_folder)

            self.__curr_save += 1

        self.__curr_iteration = (self.__curr_iteration + 1) % self.__save_every
