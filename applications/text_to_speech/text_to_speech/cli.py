from .tts.wrapped_api import (
    prepare_embedding,
    embedder as _embedder,
    text_to_speech as _text_to_speech,
    text_to_speech_generate,
    train as _train,
    training,
)
from python_utilities.logger import setup_logging
from . import config
import sounddevice
import argparse
import logging
import soundfile
import time

default_config = config.TextToSpeechConfig().default


# poetry run cli train "../../resources/training_data/tracer.mp3" "../../resources/models/openvoice/embeddings/" "tracer2"
def train(reference_file, target_directory, name):
    logging.getLogger(__name__).info(
        f"Training with reference: {reference_file}, target directory: {target_directory}, name: {name}"
    )

    trainer = training()
    _train(
        trainer,
        reference_file=reference_file,
        target_directory=target_directory,
        name=name,
    )


# poetry run cli generate --use-embedding --play "A big brown for has jumped over a lazy dog."
def generate(text, use_embedding, play, output_file):
    logging.getLogger(__name__).info(
        f"Generated content for: {text} with embedding={use_embedding} and play={play}"
    )

    if not (play or output_file):
        raise Exception("Error: Either --play or --output_file must be provided.")

    text_to_speech = _text_to_speech()

    sound_file_buffer, sampling_rate = text_to_speech_generate(
        text_to_speech,
        text,
    )

    if use_embedding:
        embedder = _embedder()
        embedding = prepare_embedding(embedder=embedder)
        sound_file_buffer, sampling_rate = embedding(sound_file_buffer, sampling_rate)

    sound_file = soundfile.read(sound_file_buffer)

    if play:
        sounddevice.play(sound_file, sampling_rate)
        sounddevice.wait()

    if output_file:
        soundfile.write(output_file, sound_file, sampling_rate)
        print(f"Output written to {output_file}")


def main():
    setup_logging(default_config.log_level())
    parser = argparse.ArgumentParser(description="CLI for training and generating.")

    subparsers = parser.add_subparsers(dest="command")

    # fmt: off
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('reference_file', type=str, help='Path to the reference file')
    train_parser.add_argument('target_directory', type=str, help='Directory to save results')
    train_parser.add_argument('name', type=str, help='Name of the training session')

    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate content')
    generate_parser.add_argument('text', type=str, nargs='?', help='Text to generate content for', default=None)
    generate_parser.add_argument('-e', '--use-embedding', action='store_true', default=False, help='Use embedding in generation')
    generate_parser.add_argument('-p', '--play', action='store_true', default=True, help='Play generated content')
    generate_parser.add_argument('-o', '--output-file', type=str, help='Path to save the output file', default=None)
    # fmt: on

    args, unknown = parser.parse_known_args()

    if args.command == "train":
        train(args.reference_file, args.target_directory, args.name)
    elif args.command == "generate":
        # Handle `text` input, either from args.text or unknown positional arguments
        text = args.text or (unknown[0] if unknown else None)
        if not text:
            print("Error: Text is required for generation.")
            return
        generate(args.text, args.use_embedding, args.play, args.output_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
