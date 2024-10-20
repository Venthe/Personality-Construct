import logging
from melo.api import TTS
import soundfile
from melo.download_utils import LANG_TO_HF_REPO_ID
from typing import TypedDict, Optional
from .utilities import to_soundfile


class TTSKwargs(TypedDict, total=False):
    speed: float
    sdp_ratio: float
    noise_scale: float
    noise_scale_w: float
    pbar: Optional[bool]
    position: Optional[int]
    quiet: bool


class TextToSpeech:
    def __init__(self, device="cpu", **kwargs):
        self.__logger = logging.getLogger(__name__)
        self.__device = device
        self.__language_model = kwargs.get("language_model", "EN_NEWEST")
        self.__speaker_key = kwargs.get("speaker_key", "EN-Newest")
        self.__text_to_speech = self.__init_text_to_speech()

    def __init_text_to_speech(self):
        self.__logger.info(f"Initiating TTS model {self.__language_model}")
        try:
            tts_model = TTS(
                language=self.__language_model.upper(), device=self.__device
            )
            self.__logger.info(f"TTS model initiated")
        except AssertionError as e:
            self.__logger.error(
                f"Language not supported, please use either of: {', '.join(list(LANG_TO_HF_REPO_ID.keys()))}"
            )
            raise

        speaker_ids = tts_model.hps.data.spk2id
        self.__logger.debug(f"Available speakers: {speaker_ids}")
        speaker_id = self.__get_value_from_suffix(speaker_ids, self.__speaker_key)

        def text_to_speech(text, **kwargs):
            return (
                tts_model.tts_to_file(
                    text,
                    speaker_id=speaker_id,
                    speed=kwargs.get("speed", 1.0),
                    sdp_ratio=kwargs.get("sdp_ratio", 0.2),
                    noise_scale=kwargs.get("noise_scale", 0.6),
                    noise_scale_w=kwargs.get("noise_scale_w", 0.8),
                    pbar=kwargs.get("pbar", None),
                    position=kwargs.get("position", None),
                    quiet=kwargs.get("quiet", False),
                ),
                tts_model.hps.data.sampling_rate,
            )

        return text_to_speech

    def __get_value_from_suffix(self, data, text):
        text_lower = text.lower()
        for key in data.__dict__:
            if text_lower == key.lower():
                return data[key]
        raise KeyError(
            f"No matching key suffix found for '{text}', {', '.join(list(data.keys()))}"
        )

    def generate(self, text, **kwargs: TTSKwargs):
        audio_data, sampling_rate = self.__text_to_speech(text, **kwargs)
        result, sampling_rate = to_soundfile(audio_data, sampling_rate=sampling_rate)
        return result, sampling_rate
