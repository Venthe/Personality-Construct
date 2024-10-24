from text_to_speech.tone_converter.tone_converter_trainer import ToneConverterTrainer
from text_to_speech.config import TextToSpeechConfiguration
from text_to_speech.tone_converter.tone_converter import (
    ToneColorConverter,
    ToneColorConverterWrapper,
)

tone_converter_config = TextToSpeechConfiguration().tone_converter
tone_converter_embedding_config = TextToSpeechConfiguration().tone_converter_embedding
tone_converter_training_config = TextToSpeechConfiguration().tone_converter_training


def create_tone_converter_trainer():
    return ToneConverterTrainer(
        tone_color_converter=ToneColorConverterWrapper(
            model_name=tone_converter_config.model_name(),
            model_path=tone_converter_config.model_path(),
            use_gpu=tone_converter_training_config.use_gpu(),
        ),
    )


def train_tone_converter(
    trainer: ToneConverterTrainer, reference_file, target_directory, name
):
    return trainer.train(
        reference_file=reference_file,
        target_directory=target_directory,
        name=name,
        use_vad=tone_converter_training_config.use_vad(),
    )


def create_tone_converter() -> ToneColorConverter:
    return ToneColorConverter(
        tone_color_converter=ToneColorConverterWrapper(
            model_name=tone_converter_config.model_name(),
            model_path=tone_converter_config.model_path(),
            use_gpu=tone_converter_config.use_gpu(),
        ),
        speaker_model_name=tone_converter_embedding_config.speaker_model_name(),
        speaker_model_path=tone_converter_embedding_config.speaker_model_path(),
        speaker_model_file=tone_converter_embedding_config.speaker_model_file(),
        embedding_checkpoint_path=tone_converter_embedding_config.embedding_checkpoint_path(),
    )


def tone_converter_process(tone_converter: ToneColorConverter, buffer):
    return tone_converter.process(
        buffer=buffer, tau=tone_converter_embedding_config.temperature()
    )
