from text_to_speech.text_to_speech.text_to_speech import TextToSpeech
from text_to_speech.config import TextToSpeechConfiguration

text_to_speech_config = TextToSpeechConfiguration().text_to_speech


def create_text_to_speech() -> TextToSpeech:
    return TextToSpeech(
        use_gpu=text_to_speech_config.use_gpu(),
        model_name=text_to_speech_config.model_name(),
        model_path=text_to_speech_config.model_path(),
        language=text_to_speech_config.language(),
        speaker_key=text_to_speech_config.speaker_key(),
    )


def text_to_speech_generate(text_to_speech: TextToSpeech, text):
    return text_to_speech.generate(
        text,
        speed=text_to_speech_config.speed(),
        sdp_ratio=text_to_speech_config.sdp_ratio(),
        noise_scale=text_to_speech_config.noise_scale(),
        noise_scale_w=text_to_speech_config.noise_scale_w(),
        quiet=text_to_speech_config.quiet(),
    )
