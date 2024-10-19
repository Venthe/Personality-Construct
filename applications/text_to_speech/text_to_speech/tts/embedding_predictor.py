import torch
import soundfile
import librosa
from .base_predictor import BasePredictor, TTSKwargs


class EmbeddingPredictor(BasePredictor):
    def __init__(
        self,
        create_tone_converter_callback,
        speaker_model,
        embedding_model,
        device="cpu",
        **kwargs
    ):
        super().__init__(device=device, **kwargs)
        self.__device = device
        self.__tone_converter = create_tone_converter_callback()
        self.__tone_converter_sampling_rate = (
            self.__tone_converter.hps.data.sampling_rate
        )
        self.__tone_convert = self.__init_tone_convert(speaker_model, embedding_model)

    def __init_tone_convert(self, speaker_model, embedding_model):
        speaker_model = torch.load(
            speaker_model, map_location=torch.device(self.__device)
        )
        embedding_model = torch.load(
            embedding_model, map_location=torch.device(self.__device)
        )

        def tone_convert(audio, tau=0.2):
            return self.__tone_converter.convert(
                audio_src_path=audio,
                src_se=speaker_model,
                tgt_se=embedding_model,
                tau=tau,
            )

        return tone_convert

    def __apply_embedding(self, result, sampling_rate, tau=0.2):
        result, _ = self.__resample_audio(result, self.__tone_converter_sampling_rate)
        tc = self.__tone_convert(result, tau=tau)
        result, sampling_rate = self._BasePredictor__to_soundfile(
            tc, self.__tone_converter_sampling_rate
        )
        return result, sampling_rate

    def __resample_audio(self, audio_buffer, target_sr=44100):
        audio_buffer.seek(0)
        audio_data, original_sampling_rate = soundfile.read(audio_buffer)

        # Resample the audio data
        resampled_audio = librosa.resample(
            audio_data, orig_sr=original_sampling_rate, target_sr=target_sr
        )

        return self._BasePredictor__to_soundfile(resampled_audio, target_sr)

    def convert(self, text, tau=0.3, **kwargs: TTSKwargs):
        result, sampling_rate = self._BasePredictor__tts(text, **kwargs)
        result, sampling_rate = self.__apply_embedding(result, sampling_rate, tau)
        sound_file, _ = soundfile.read(result)
        return sound_file, sampling_rate
