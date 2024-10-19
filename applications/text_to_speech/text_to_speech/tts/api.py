from python_utilities.cuda import detect_cuda
from .trainer import Trainer
from .base_predictor import BasePredictor
from .embedding_predictor import EmbeddingPredictor
from .tone_converter import create_tone_converter


__default_converter_path = "model/converter/"


def training(converter_path=__default_converter_path, use_gpu=True):
    return Trainer(
        create_tone_converter_callback=lambda: create_tone_converter(
            device=__select_device(use_gpu), converter_path=converter_path
        )
    )


def prediction(use_gpu=True, **kwargs):
    return BasePredictor(device=__select_device(use_gpu), **kwargs)


def prediction_with_embedding(
    embedding_model,
    speaker_model = "model/base_speakers/ses/en-default.pth",
    converter_path=__default_converter_path,
    use_gpu=True,
    **kwargs
):
    return EmbeddingPredictor(
        device=__select_device(use_gpu),
        speaker_model=speaker_model,
        embedding_model=embedding_model,
        create_tone_converter_callback=lambda: create_tone_converter(
            device=__select_device(use_gpu), converter_path=converter_path
        ),
        **kwargs
    )


def __select_device(use_gpu=True):
    return "cuda:0" if use_gpu and detect_cuda() else "cpu"
