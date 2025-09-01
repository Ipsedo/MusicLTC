import argparse

from .data.create_dataset import create_dataset


def main() -> None:
    parser = argparse.ArgumentParser("music_ltc main")

    sub_parsers = parser.add_subparsers(required=True, dest="mode")

    create_dataset_parser = sub_parsers.add_parser(name="create-dataset")
    create_dataset_parser.add_argument("audio_glob_path", type=str)
    create_dataset_parser.add_argument("--output-dir", type=str, required=True)
    create_dataset_parser.add_argument(
        "--sample-rate", type=int, default=16000
    )
    create_dataset_parser.add_argument(
        "--sequence-length", type=int, default=160000
    )

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_dataset(
            args.audio_glob_path,
            args.output_dir,
            args.sample_rate,
            args.sequence_length,
        )
    else:
        parser.error(f"Invalid mode : {args.mode}")


if __name__ == "__main__":
    main()
