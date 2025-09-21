from pydantic import BaseModel

from .networks.denoiser import Denoiser
from .networks.noiser import Noiser


class ModelOptions(BaseModel):
    diffusion_steps: int
    time_size: int
    channels: list[tuple[int, int]]
    neuron_number: int
    unfolding_steps: int
    delta_t: float

    def get_noiser(self) -> Noiser:
        return Noiser(self.diffusion_steps)

    def get_denoiser(self) -> Denoiser:
        return Denoiser(
            self.diffusion_steps,
            self.time_size,
            self.channels,
            self.neuron_number,
            self.unfolding_steps,
            self.delta_t,
        )


class TrainOptions(BaseModel):
    dataset_path: str
    output_dir: str
    batch_size: int
    epochs: int
    learning_rate: float
    gamma: float
    sample_rate: int
    nb_audios_to_generate: int
    audios_to_generate_length: int
    fast_sample_steps: int
    cuda: bool
    dataloader_workers: int
    save_every: int
