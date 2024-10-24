import os
import logging
from openvoice.api import ToneColorConverter as _ToneColorConverter
from text_to_speech.utilities.audio import resample_audio, to_soundfile
from text_to_speech.utilities.device import select_device
import torch

from _utilities.models import download_model_if_empty


class ToneColorConverterWrapper:
    model: _ToneColorConverter
    model_name: str
    model_path: str

    def __init__(self, model_path, model_name, use_gpu=True):
        logger = logging.getLogger(__name__)

        self.model_name = model_name
        self.model_path = model_path

        download_model_if_empty(model_path, model_name)

        logger.debug(f"Creating tone converter for path {model_path}")
        self.model = self._load_model(model_path, model_name, use_gpu)
        logger.info(f"Tone converter created for path {model_path}")

    def _load_model(self, model_path, model_name, use_gpu):
        device = select_device(use_gpu)

        config_file = os.path.normpath(
            os.path.join(model_path, model_name, "converter", "config.json")
        )
        model = _ToneColorConverter(config_path=config_file, device=device)
        checkpoint_file = os.path.normpath(
            os.path.join(model_path, model_name, "converter", "checkpoint.pth")
        )
        model.load_ckpt(checkpoint_file)
        return model


class ToneColorConverter:
    _embedding_model = None

    def __init__(
        self,
        tone_color_converter: ToneColorConverterWrapper,
        speaker_model_path,
        speaker_model_name,
        speaker_model_file,
        embedding_checkpoint_path,
    ):
        self._logger = logging.getLogger(__name__)
        self._tone_converter = tone_color_converter.model
        self._device = self._tone_converter.device
        self._sampling_rate = self._tone_converter.hps.data.sampling_rate
        self._speaker_model = self._load_speaker_model(speaker_model_path, speaker_model_name, speaker_model_file)
        self._embedding_model = self.load_embedding_model(embedding_checkpoint_path)

    def _load_speaker_model(self, model_path, model_name, speaker_model_file):
        download_model_if_empty(model_path, model_name)

        return torch.load(
            os.path.join(model_path, model_name, "base_speakers", "ses", speaker_model_file), map_location=torch.device(self._device)
        )

    def load_embedding_model(self, embedding_checkpoint_path):
        return torch.load(
            embedding_checkpoint_path,
            map_location=torch.device(self._device),
        )

    def process(self, buffer, tau=0.2):
        result, _ = resample_audio(buffer, self._sampling_rate)
        tc = self._tone_converter.convert(
            audio_src_path=result,
            src_se=self._speaker_model,
            tgt_se=self._embedding_model,
            tau=tau,
        )
        result, sampling_rate = to_soundfile(tc, self._sampling_rate)
        return result, sampling_rate
