from text_to_speech.tone_converter.default_api import (
    create_tone_converter_trainer,
    train_tone_converter,
    create_tone_converter,
    tone_converter_process,
)
from text_to_speech.text_to_speech.default_api import (
    create_text_to_speech,
    text_to_speech_generate,
)
from _utilities.logging import setup_logging
from . import config
import sounddevice
import argparse
import logging
import soundfile

default_config = config.TextToSpeechConfiguration().default


# poetry run cli generate --convert-tone --play "A big brown fox has jumped over a lazy dog." --config-override=section=openvoice-embedding,option=embedding_model,value=../../resources/models/openvoice/embeddings/dva/checkpoint.pth
def generate(text, convert_tone, play, output_file):
    logging.getLogger(__name__).info(
        f"Generated content for: {text} with convert_tone={convert_tone} and play={play}"
    )

    if not (play or output_file):
        raise Exception("Error: Either --play and/or --output_file must be provided.")

    text_to_speech = create_text_to_speech()

    sound_file_buffer, sampling_rate = text_to_speech_generate(
        text_to_speech,
        text,
    )

    if convert_tone:
        tone_converter = create_tone_converter()
        sound_file_buffer, sampling_rate = tone_converter_process(tone_converter=tone_converter, buffer=sound_file_buffer)

    sound_file, _ = soundfile.read(sound_file_buffer)

    if play:
        sounddevice.play(sound_file, sampling_rate)
        sounddevice.wait()

    if output_file:
        soundfile.write(output_file, sound_file, sampling_rate)
        print(f"Output written to {output_file}")


# poetry run cli train "../../resources/training_data/tracer.mp3" "../../resources/embeddings/OpenVoiceV2/" "tracer"
def train(reference_file, target_directory, name):
    logging.getLogger(__name__).info(
        f"Training with reference: {reference_file}, target directory: {target_directory}, name: {name}"
    )

    trainer = create_tone_converter_trainer()
    train_tone_converter(
        trainer,
        reference_file=reference_file,
        target_directory=target_directory,
        name=name,
    )


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
    generate_parser.add_argument('-c', '--convert-tone', action='store_true', default=False, help='Convert tone after generation')
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
        generate(args.text, args.convert_tone, args.play, args.output_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
