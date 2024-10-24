from _utilities.logging import map_log_level
from _configuration.utilities import parse_bool
from _configuration import Configuration


# class OpenVoiceEmbedding:
#     def __init__(self, data):
#         self.data = data

#     def speaker_model(self):
#         return self.data.get("speaker_model")

#     def embedding_model(self):
#         return self.data.get("embedding_model")

#     def tau(self):
#         return float(self.data.get("tau"))

#     def converter_path(self):
#         return self.data.get("converter_path")


# class OpenVoice:
#     def __init__(self, data):
#         self.data = data

#     def use_gpu(self):
#         return parse_bool(self.data.get("use_gpu"))

#     def embedding_path(self):
#         return self.data.get("embedding_path")

#     def language_model(self):
#         return self.data.get("language_model")

#     def speaker_key(self):
#         return self.data.get("speaker_key")


# class OpenVoiceEmbeddingTraining:
#     def __init__(self, data):
#         self.data = data

#     def use_vad(self):
#         return parse_bool(self.data.get("use_vad"))


class Default:
    def __init__(self, data):
        self.data = data

    def log_level(self):
        return map_log_level(self.data.get("log_level"))

    def log_path(self):
        return self.data.get("log_path") if self.data.get("log_path") else None


class TextToSpeech:
    def __init__(self, data):
        self.data = data

    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))

    def model_name(self):
        return self.data.get("model_name")

    def model_path(self):
        return self.data.get("model_path")

    def language(self):
        return self.data.get("language")

    def speaker_key(self):
        return self.data.get("speaker_key")

    def speed(self):
        return float(self.data.get("speed"))

    def sdp_ratio(self):
        return float(self.data.get("sdp_ratio"))

    def noise_scale(self):
        return float(self.data.get("noise_scale"))

    def noise_scale_w(self):
        return float(self.data.get("noise_scale_w"))

    def quiet(self):
        return parse_bool(self.data.get("quiet"))


class ToneConverter:
    def __init__(self, data):
        self.data = data

    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))

    def model_path(self):
        return self.data.get("model_path")

    def model_name(self):
        return self.data.get("model_name")


class ToneConverterEmbedding:
    def __init__(self, data):
        self.data = data

    def speaker_model_path(self):
        return self.data.get("speaker_model_path")

    def speaker_model_name(self):
        return self.data.get("speaker_model_name")

    def speaker_model_file(self):
        return self.data.get("speaker_model_file")

    def embedding_checkpoint_path(self):
        return self.data.get("embedding_checkpoint_path")

    def temperature(self):
        return float(self.data.get("temperature"))


class ToneConverterTraining:
    def __init__(self, data):
        self.data = data

    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))

    def use_vad(self):
        return parse_bool(self.data.get("use_vad"))


class Server:
    def __init__(self, data):
        self.data = data

    def port(self):
        return int(self.data.get("port"))

    def host(self):
        return self.data.get("host")

    def debug(self):
        return parse_bool(self.data.get("debug"))


class TextToSpeechConfiguration(Configuration):
    def __init__(self):
        super().__init__()
        self.default = Default(self.config["default"])
        self.text_to_speech = TextToSpeech(self.config["text-to-speech"])
        self.tone_converter = ToneConverter(self.config["tone-converter"])
        self.tone_converter_embedding = ToneConverterEmbedding(
            self.config["tone-converter.embedding"]
        )
        self.tone_converter_training = ToneConverterTraining(
            self.config["tone-converter.training"]
        )
        self.server = Server(self.config["server"])

        # self.openvoice = OpenVoice(self.config["openvoice1"])
        # self.embedding = OpenVoiceEmbedding(self.config["openvoice-embedding1"])
        # self.training = OpenVoiceEmbeddingTraining(
        #     self.config["openvoice-embedding-training1"]
        # )
