import glob
from os import mkdir
from os.path import exists, isdir, join

import torch as th
from tqdm import tqdm

from .audio import open_audio_and_convert_sample_rate, split_audio


def create_dataset(
    audio_path: str,
    dataset_output_dir: str,
    sample_rate: int,
    sequence_length: int,
) -> None:
    w_p = glob.glob(audio_path, recursive=True)

    if not exists(dataset_output_dir):
        mkdir(dataset_output_dir)
    elif not isdir(dataset_output_dir):
        raise NotADirectoryError(dataset_output_dir)

    idx = 0

    tqdm_bar = tqdm(w_p)

    for wav_p in tqdm_bar:
        waveform_tensor = open_audio_and_convert_sample_rate(
            wav_p, sample_rate
        )

        split_waveforms = split_audio(waveform_tensor, sequence_length)

        for sample in split_waveforms:
            output_file_path = join(dataset_output_dir, f"waveform_{idx}.pt")

            th.save(sample.transpose(0, 1).clone(), output_file_path)

            idx += 1

        tqdm_bar.set_description(f"total : {idx}")
