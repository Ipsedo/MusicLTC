import torch as th
import torchaudio as th_audio


def open_audio_and_convert_sample_rate(
    audio_path: str,
    to_sample_rate: int,
) -> th.Tensor:
    raw_audio, sr = th_audio.load_with_torchcodec(audio_path)

    resampled_raw_audio: th.Tensor = th_audio.functional.resample(raw_audio, sr, to_sample_rate)
    return resampled_raw_audio


def split_audio(waveform_tensor: th.Tensor, sequence_length: int) -> list[th.Tensor]:
    return list(th.split(waveform_tensor, sequence_length, 1))[:-1]
