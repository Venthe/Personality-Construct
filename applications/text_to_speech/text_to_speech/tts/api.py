from python_utilities.cuda import detect_cuda
from .text_to_speech import TextToSpeech
from .embedding import Embedder, Trainer
from .tone_converter import create_tone_converter


__default_converter_path = "model/converter/"


def training(converter_path=__default_converter_path, use_gpu=True):
    return Trainer(
        create_tone_converter_callback=lambda: create_tone_converter(
            device=__select_device(use_gpu), converter_path=converter_path
        )
    )


def text_to_speech(use_gpu=True, **kwargs):
    return TextToSpeech(device=__select_device(use_gpu), **kwargs)


def embedder(
    embedding_model,
    speaker_model = "model/base_speakers/ses/en-default.pth",
    converter_path=__default_converter_path,
    use_gpu=True
):
    return Embedder(
        device=__select_device(use_gpu),
        speaker_model=speaker_model,
        embedding_model=embedding_model,
        create_tone_converter_callback=lambda: create_tone_converter(
            device=__select_device(use_gpu), converter_path=converter_path
        )
    )


def __select_device(use_gpu=True):
    return "cuda:0" if use_gpu and detect_cuda() else "cpu"
