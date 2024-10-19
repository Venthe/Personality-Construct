import logging
from melo.api import TTS
import soundfile
import io
from melo.download_utils import LANG_TO_HF_REPO_ID
from typing import TypedDict, Optional

class TTSKwargs(TypedDict, total=False):
    speed: float
    sdp_ratio: float
    noise_scale: float
    noise_scale_w: float
    pbar: Optional[bool]
    position: Optional[int]
    quiet: bool

class BasePredictor:
    def __init__(
        self,
        config,
        device="cpu",
    ):
        self.__logger = logging.getLogger(__name__)
        self.__device = device
        self.__config = config
        self.__text_to_speech = self.__init_text_to_speech()

    def __init_text_to_speech(self):
        try:
            tts_model = TTS(
                language=self.__config.language_model().upper(), device=self.__device
            )
        except AssertionError as e:
            self.__logger.error(
                f"Language not supported, please use either of: {', '.join(list(LANG_TO_HF_REPO_ID.keys()))}"
            )
            raise

        speaker_ids = tts_model.hps.data.spk2id
        self.__logger.debug(f"Available speakers: {speaker_ids}")
        speaker_id = self.__get_value_from_suffix(
            speaker_ids, self.__config.speaker_key()
        )

        def text_to_speech(
            text, **kwargs
        ):
            return (
                tts_model.tts_to_file(
                    text,
                    speed=kwargs.get("speed", 1.0),
                    speaker_id=speaker_id,
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
            suffix = key.split("-")[-1].lower()
            if text_lower == suffix:
                return data[key]
        raise KeyError(
            f"No matching key suffix found for '{text}', {', '.join(list(data.keys()))}"
        )

    def __to_soundfile(self, audio_data, sampling_rate=44100):
        audio_buffer = io.BytesIO()
        soundfile.write(audio_buffer, audio_data, sampling_rate, format="WAV")
        audio_buffer.seek(0)

        return audio_buffer, sampling_rate

    def __tts(
        self,
        text,
        **kwargs
    ):
        audio_data, sampling_rate = self.__text_to_speech(
            text,
            **kwargs
        )
        result, sampling_rate = self.__to_soundfile(
            audio_data, sampling_rate=sampling_rate
        )
        return result, sampling_rate

    def convert(
        self,
        text,
        **kwargs: TTSKwargs
    ):
        """
        Convert text to speech with optional parameters:
        speed (default=1.0), sdp_ratio (default=0.2), noise_scale (default=0.6),
        noise_scale_w (default=0.8), pbar, position, quiet (default=False)
        """
        result, sampling_rate = self.__tts(
            text,
            **kwargs
        )
        sound_file, _ = soundfile.read(result)
        return sound_file, sampling_rate
