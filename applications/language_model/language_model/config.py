from _configuration import Configuration
from _utilities.logging import map_log_level


class LLM:
    def __init__(self, data):
        self.data = data

    def model_path(self):
        return self.data.get("model_path")

    def model_name(self):
        return self.data.get("model_name")


class Default:
    def __init__(self, data):
        self.data = data

    def log_level(self):
        return map_log_level(self.data.get("log_level"))


class LanguageModelConfig(Configuration):
    def __init__(self):
        super().__init__()
        self.llm = LLM(self.config["llm"])
        self.default = Default(self.config["default"])
