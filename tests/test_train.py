import tempfile
from os import mkdir
from os.path import exists, isdir, isfile, join

import torch as th

from music_ltc.options import ModelOptions, TrainOptions
from music_ltc.train import train_model


def test_train_model() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = join(tmpdir, "output")
        dataset_dir = join(tmpdir, "dataset")

        channels = 2
        nb_audio_to_generate = 1
        audio_length = 64

        single_data = th.randn(audio_length, channels)

        mkdir(dataset_dir)
        th.save(single_data, join(dataset_dir, "waveform_0.pt"))

        model_options = ModelOptions(
            diffusion_steps=4,
            time_size=2,
            channels=channels,
            hidden_channels=[(2, 2)],
            neuron_number=4,
            unfolding_steps=2,
            delta_t=1.0,
        )

        train_options = TrainOptions(
            dataset_path=dataset_dir,
            output_dir=output_dir,
            batch_size=1,
            epochs=1,
            learning_rate=1e-4,
            gamma=0.1,
            sample_rate=16000,
            nb_audios_to_generate=nb_audio_to_generate,
            audios_to_generate_length=audio_length,
            fast_sample_steps=2,
            cuda=False,
            dataloader_workers=1,
            save_every=1,
        )

        train_model(model_options, train_options)

        first_save_path = join(output_dir, "save_0")
        assert exists(first_save_path)
        assert isdir(first_save_path)

        assert exists(join(first_save_path, "noiser.pth"))
        assert isfile(join(first_save_path, "noiser.pth"))

        assert exists(join(first_save_path, "denoiser.pth"))
        assert isfile(join(first_save_path, "denoiser.pth"))

        assert exists(join(first_save_path, "optimizer.pth"))
        assert isfile(join(first_save_path, "optimizer.pth"))

        assert exists(join(first_save_path, "waveform.pt"))
        assert isfile(join(first_save_path, "waveform.pt"))

        x_0 = th.load(join(first_save_path, "waveform.pt"))
        assert x_0.size() == (nb_audio_to_generate, audio_length, channels)

        assert exists(join(first_save_path, "audio_0.wav"))
        assert isfile(join(first_save_path, "audio_0.wav"))
