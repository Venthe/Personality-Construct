from python_config.config import Config
from python_config.utilities import parse_bool
from python_utilities.utilities import map_log_level


class OpenVoiceEmbedding:
    def __init__(self, data):
        self.data = data

    def speaker_model(self):
        return self.data.get("speaker_model")

    def embedding_model(self):
        return self.data.get("embedding_model")

    def tau(self):
        return float(self.data.get("tau"))

    def converter_path(self):
        return self.data.get("converter_path")


class OpenVoice:
    def __init__(self, data):
        self.data = data

    def use_gpu(self):
        return parse_bool(self.data.get("use_gpu"))

    def embedding_path(self):
        return self.data.get("embedding_path")

    def language_model(self):
        return self.data.get("language_model")

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


class Default:
    def __init__(self, data):
        self.data = data

    def log_level(self):
        return map_log_level(self.data.get("log_level"))

    def log_path(self):
        return self.data.get("log_path") if self.data.get("log_path") else None


class Server:
    def __init__(self, data):
        self.data = data

    def port(self):
        return int(self.data.get("port"))

    def host(self):
        return self.data.get("host")

    def debug(self):
        return parse_bool(self.data.get("debug"))


class OpenVoiceEmbeddingTraining:
    def __init__(self, data):
        self.data = data

    def use_vad(self):
        return parse_bool(self.data.get("use_vad"))


class TextToSpeechConfig(Config):
    def __init__(self):
        super().__init__()
        self.openvoice = OpenVoice(self.config["openvoice"])
        self.embedding = OpenVoiceEmbedding(self.config["openvoice-embedding"])
        self.training = OpenVoiceEmbeddingTraining(
            self.config["openvoice-embedding-training"]
        )
        self.server = Server(self.config["server"])
        self.default = Default(self.config["default"])
