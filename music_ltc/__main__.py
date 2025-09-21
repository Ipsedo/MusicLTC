import argparse
import re

from .data.create_dataset import create_dataset
from .options import ModelOptions, TrainOptions
from .train import train_model


def _channels(string: str) -> list[tuple[int, int]]:
    regex_match = re.compile(r"^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$")
    regex_layer = re.compile(r"\( *\d+ *, *\d+ *\)")
    regex_channel = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [(10, 20), (20, 40), ...]"

    def _match_channels(layer_str: str) -> tuple[int, int]:
        matched = regex_channel.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_channels(layer) for layer in regex_layer.findall(string)]


def main() -> None:
    parser = argparse.ArgumentParser("music_ltc main")

    sub_parsers = parser.add_subparsers(required=True, dest="mode")

    create_dataset_parser = sub_parsers.add_parser(name="create-dataset")
    create_dataset_parser.add_argument("audio_glob_path", type=str)
    create_dataset_parser.add_argument("--output-dir", type=str, required=True)
    create_dataset_parser.add_argument("--sample-rate", type=int, default=16000)
    create_dataset_parser.add_argument("--sequence-length", type=int, default=2**17)

    model_parser = sub_parsers.add_parser(name="model")
    model_parser.add_argument("--diffusion-steps", type=int, default=4096)
    model_parser.add_argument("--time-size", type=int, default=32)
    model_parser.add_argument(
        "--channels", type=_channels, default=[(2, 32), (32, 64), (64, 128), (128, 256)]
    )
    model_parser.add_argument("--neuron-number", type=int, default=128)
    model_parser.add_argument("--unfolding-steps", type=int, default=6)
    model_parser.add_argument("--delta-t", type=float, default=1.6e-2)

    model_parser_subparser = model_parser.add_subparsers(dest="run", required=True)

    train_model_parser = model_parser_subparser.add_parser(name="train")
    train_model_parser.add_argument("dataset_path", type=str)
    train_model_parser.add_argument("-o", "--output-dir", type=str, required=True)
    train_model_parser.add_argument("--batch-size", type=int, default=20)
    train_model_parser.add_argument("--epochs", type=int, default=1000)
    train_model_parser.add_argument("--learning-rate", type=float, default=1e-3)
    train_model_parser.add_argument("--gamma", type=float, default=1e-2)
    train_model_parser.add_argument("--sample-rate", type=int, default=16000)
    train_model_parser.add_argument("--fast-sample-steps", type=int, default=256)
    train_model_parser.add_argument("--nb-audios-to-generate", type=int, default=3)
    train_model_parser.add_argument("--audios-to-generate-length", type=int, default=2**17)
    train_model_parser.add_argument("--cuda", action="store_true")
    train_model_parser.add_argument("--save-every", type=int, default=2048)
    train_model_parser.add_argument("--dataloader-workers", type=int, default=8)

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_dataset(
            args.audio_glob_path, args.output_dir, args.sample_rate, args.sequence_length
        )

    elif args.mode == "model":
        model_options = ModelOptions(
            diffusion_steps=args.diffusion_steps,
            time_size=args.time_size,
            channels=args.channels,
            neuron_number=args.neuron_number,
            unfolding_steps=args.unfolding_steps,
            delta_t=args.delta_t,
        )

        if args.run == "train":
            train_model(
                model_options,
                TrainOptions(
                    dataset_path=args.dataset_path,
                    output_dir=args.output_dir,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    learning_rate=args.learning_rate,
                    gamma=args.gamma,
                    fast_sample_steps=args.fast_sample_steps,
                    sample_rate=args.sample_rate,
                    nb_audios_to_generate=args.nb_audios_to_generate,
                    audios_to_generate_length=args.audios_to_generate_length,
                    cuda=args.cuda,
                    dataloader_workers=args.dataloader_workers,
                    save_every=args.save_every,
                ),
            )

        else:
            model_parser.error(f"Invalid run : {args.run}")

    else:
        parser.error(f"Invalid mode : {args.mode}")


if __name__ == "__main__":
    main()
