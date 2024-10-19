from .tts.wrapped_api import (
    prepare_embedding,
    embedder as _embedder,
    text_to_speech as _text_to_speech,
    text_to_speech_generate,
    training,
    train,
)
from python_utilities.logger import setup_logging
from . import config
import sounddevice

default_config = config.TextToSpeechConfig().default


def speak():
    setup_logging(default_config.log_level())

    # trainer = training()
    # train(
    #     trainer,
    #     reference_file="../../resources/training_data/tracer.mp3",
    #     target_directory="../../resources/models/openvoice/embeddings/",
    #     name="tracer",
    # )
    # del trainer

    text_to_speech = _text_to_speech()

    wav, sampling_rate = text_to_speech_generate(
        text_to_speech, "Cheese is here. What do you want to say, love???"
    )
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()

    embedder = _embedder()
    wav, sampling_rate = text_to_speech_generate(
        text_to_speech,
        "Cheese is here. What do you want to say, love???",
        prepare_embedding(embedder=embedder),
    )
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()
