import logging
from melo.api import TTS
from typing import TypedDict, Optional

from text_to_speech.utilities.audio import to_soundfile
from text_to_speech.utilities.device import select_device
from collections.abc import Mapping
import os

from _utilities.models import download_model_if_empty


class TTSKwargs(TypedDict, total=False):
    speed: float
    sdp_ratio: float
    noise_scale: float
    noise_scale_w: float
    pbar: Optional[bool]
    position: Optional[int]
    quiet: bool


class TextToSpeech:
    def __init__(
        self,
        use_gpu: bool = False,
        model_name="myshell-ai/MeloTTS-English-v3",
        model_path="models",
        language=None,
        speaker_key=None,
    ):
        self._logger = logging.getLogger(__name__)
        # self._text_to_speech: Callable = self.__init_text_to_speech()
        self._model: TTS = self._init_model(
            model_path=model_path,
            model_name=model_name,
            use_gpu=use_gpu,
            language=language,
        )
        self._speaker_id = self._get_speaker_id(speaker_key=speaker_key)

    def _init_model(self, model_path, model_name, use_gpu, language):
        download_model_if_empty(model_path, model_name)

        device: str = select_device(use_gpu=use_gpu)
        self._logger.debug(f"Initiating TTS model {model_name} on {device}")
        model = TTS(
            language=language,
            device=device,
            ckpt_path=os.path.normpath(
                os.path.join(model_path, model_name, "checkpoint.pth")
            ),
            config_path=os.path.normpath(
                os.path.join(model_path, model_name, "config.json")
            ),
        )
        self._logger.info(f"TTS model initiated")
        return model

    def _get_speaker_id(self, speaker_key):
        speaker_ids = self._model.hps.data.spk2id
        self._logger.debug(f"Available speakers for model: {speaker_ids}")
        return self._get_value_from_suffix(speaker_ids, speaker_key)

    def _get_value_from_suffix(self, data: Mapping[str, str], suffix: str):
        text_lower = suffix.lower()
        for key in data.__dict__:
            if text_lower == key.lower():
                return data[key]
        raise KeyError(
            f"No matching key suffix found for '{suffix}', {', '.join(list(data.keys()))}"
        )

    def generate(self, text: str, **kwargs: TTSKwargs):
        # TODO: Pre-download BERT
        audio_data = self._model.tts_to_file(
            text,
            speaker_id=self._speaker_id,
            speed=kwargs.get("speed", 1.0),
            sdp_ratio=kwargs.get("sdp_ratio", 0.2),
            noise_scale=kwargs.get("noise_scale", 0.6),
            noise_scale_w=kwargs.get("noise_scale_w", 0.8),
            pbar=kwargs.get("pbar", None),
            position=kwargs.get("position", None),
            quiet=kwargs.get("quiet", False),
        )
        sampling_rate = self._model.hps.data.sampling_rate
        return to_soundfile(audio_data, sampling_rate=sampling_rate)
