from .tts.wrapped_api import (
    embedding_predict,
    prediction_with_embedding,
    prediction,
    base_predict,
    training,
    train,
)
from python_utilities.logger import setup_logging
from . import config
import sounddevice

default_config = config.TextToSpeechConfig().default


def speak():
    setup_logging(default_config.log_level())

    trainer = training()
    train(
        trainer,
        reference_file="../../resources/training_data/tracer.mp3",
        target_directory="../../resources/models/openvoice/embeddings/",
        name="tracer",
    )

    predictor = prediction_with_embedding()
    wav, sampling_rate = embedding_predict(
        predictor, "Cheese is here. What do you want to say, love?"
    )
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()

    predictor = prediction()

    wav, sampling_rate = base_predict(
        predictor, "Cheese is here. What do you want to say, love?"
    )
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()
