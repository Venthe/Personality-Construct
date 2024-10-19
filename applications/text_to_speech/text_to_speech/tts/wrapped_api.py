from .. import config
from . import api

openvoice_config = config.TextToSpeechConfig().openvoice
openvoice_embedding_config = config.TextToSpeechConfig().embedding
openvoice_embedding_training_config = config.TextToSpeechConfig().training


def training():
    return api.training(
        use_gpu=openvoice_config.use_gpu(),
        converter_path=openvoice_embedding_config.converter_path(),
    )


def train(trainer, reference_file, target_directory, name):
    return trainer.train(
        reference_file=reference_file,
        target_directory=target_directory,
        name=name,
        use_vad=openvoice_embedding_training_config.use_vad(),
    )


def prediction_with_embedding():
    return api.prediction_with_embedding(
        use_gpu=openvoice_config.use_gpu(),
        converter_path=openvoice_embedding_config.converter_path(),
        speaker_model=openvoice_embedding_config.speaker_model(),
        embedding_model=openvoice_embedding_config.embedding_model(),
        language_model=openvoice_config.language_model(),
        speaker_key=openvoice_config.speaker_key(),
    )


def prediction():
    return api.prediction(
        use_gpu=openvoice_config.use_gpu(),
        language_model=openvoice_config.language_model(),
        speaker_key=openvoice_config.speaker_key(),
    )


def embedding_predict(predictor, text):
    return predictor.convert(
        text,
        tau=openvoice_embedding_config.tau(),
        speed=openvoice_config.speed(),
        sdp_ratio=openvoice_config.sdp_ratio(),
        noise_scale=openvoice_config.noise_scale(),
        noise_scale_w=openvoice_config.noise_scale_w(),
        quiet=openvoice_config.quiet(),
    )


def base_predict(predictor, text):
    return predictor.convert(
        text,
        speed=openvoice_config.speed(),
        sdp_ratio=openvoice_config.sdp_ratio(),
        noise_scale=openvoice_config.noise_scale(),
        noise_scale_w=openvoice_config.noise_scale_w(),
        quiet=openvoice_config.quiet(),
    )
