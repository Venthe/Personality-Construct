import configparser
import os
import sys
import re


class Configuration:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Configuration, cls).__new__(cls)
            cls._instance.__configuration_path = "configuration.ini"
            cls._instance.config = configparser.ConfigParser()
        return cls._instance

    def __init__(self, *args, **kwargs):
        self._instance.load_config_from_file()
        self._instance.override_file()
        # cls._instance.override_environment_variables()
        self._instance.override_arguments()

    def load_config_from_file(self):
        if self.__configuration_path:
            self.config.read(os.path.join(os.getcwd(), self.__configuration_path))

    def override_file(self):
        override = self.config.get("default", "override_path", fallback="override.ini")
        if override != None and os.path.exists(override):
            self.config.read(override)

    def override_environment_variables(self):
        prefix = "CONFIG_OVERRIDE"
        for key, value in os.environ.items():
            if not key.startswith(prefix):
                continue
            arr = [item.strip() for item in value.split(";")]
            for el in arr:
                section, option, value = extract_ini_parameter(el)

                if not (
                    self.config.has_section(section)
                    and self.config.has_option(section, option)
                ):
                    continue
                self.override_value(section, option, value)

    def override_arguments(self, arguments=None):
        args = sys.argv[1:] if arguments is None else arguments

        # Process each argument
        for index, arg in enumerate(args):
            if not arg.startswith("--config-override"):
                continue

            stripped_arg = arg.replace("--config-override", "")
            if len(stripped_arg) != 0:
                # =...
                if stripped_arg.startswith("="):
                    stripped_arg = stripped_arg[1:]
                section, option, value = extract_ini_parameter(stripped_arg)

                if not (
                    self.config.has_section(section)
                    and self.config.has_option(section, option)
                ):
                    continue
                self.override_value(section, option, value)
            else:
                try:
                    new_arg = args[index + 1]
                    section, option, value = extract_ini_parameter(new_arg)

                    if not (
                        self.config.has_section(section)
                        and self.config.has_option(section, option)
                    ):
                        continue
                    self.override_value(section, option, value)
                except:
                    continue

    def override_value(self, section, key, value):
        # Ensure the section exists in the config
        if not self.config.has_section(section):
            self.config.add_section(section)

        # Set the value in the config
        self.config.set(section, key, str(value) if value is not None else None)


def extract_ini_parameter(value: str):
    value = value.strip()
    separators = ["'", '"']

    for separator in separators:
        if value.startswith(separator) and value.endswith(separator):
            value = value[1:-1].strip()
            break

    pattern = r"^section=([^,]+),option=([^,]+),value=([^,]+)?$"
    match = re.match(pattern, value)
    if not match:
        return None, None, None

    section = match.group(1).strip()
    option = match.group(2).strip()
    value = match.group(3).strip()

    return (section, option, value)
