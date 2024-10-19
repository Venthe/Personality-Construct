from .tts.api import prediction, training, prediction_with_embedding
from python_utilities.logger import setup_logging
from .config import TextToSpeechConfig
import sounddevice


def speak():
    setup_logging(TextToSpeechConfig().default.log_level())

    openvoice_config = TextToSpeechConfig().openvoice
    openvoice_embedding_config = TextToSpeechConfig().embedding

    if False:
        trainer = training(
            use_gpu=False,  # openvoice_config.use_gpu(),
            converter_path=openvoice_config.converter_path(),
        )
        trainer.train(
            reference_file="../../resources/training_data/tracer.mp3",
            target_directory="../../resources/models/openvoice/embeddings/",
            name="tracer",
            use_vad=True,
        )

    a = 1
    if a == 1:
        predictor = prediction_with_embedding(
            use_gpu=openvoice_config.use_gpu(),
            config=openvoice_config,
            converter_path=openvoice_config.converter_path(),
            speaker_model=openvoice_embedding_config.speaker_model(),
            embedding_model=openvoice_embedding_config.embedding_model(),
        )
    elif a == 2:
        predictor = prediction(
            use_gpu=openvoice_config.use_gpu(),
            config=openvoice_config,
        )

    wav, sampling_rate = predictor.convert(
        "Cheese is here. What do you want to say, love?"
    )
    sounddevice.play(wav, sampling_rate)
    sounddevice.wait()
