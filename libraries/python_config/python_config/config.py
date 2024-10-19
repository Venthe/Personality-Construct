import configparser
import os
import sys
import re


class SingleConfig:
    def __init__(
        self, configuration_path="configuration.ini", initial_data={}, arguments=None
    ):
        self.__configuration_path = configuration_path
        self.load_config_from_file()
        self.load_initial_data(initial_data)
        self.override_file()
        # self.override_environment_variables()
        self.override_arguments(arguments)

    def load_config_from_file(self):
        config = configparser.ConfigParser()
        config.read(os.path.join(os.getcwd(), self.__configuration_path))

        self.config = config

    def load_initial_data(self, initial_data):
        for section, key_value_pairs in initial_data.items():
            for kv in key_value_pairs:
                # Extract the key and value from the dictionary
                for key, value in kv.items():
                    # Override the value for the given section and key
                    self.override_value(section, key, value)

    def override_file(self):
        override = self.config.get("default", "override_path", fallback="override.ini")
        if override != None and os.path.exists(override):
            self.config.read(override)

    def override_environment_variables(self):
        prefix = "CONFIG_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix to get section and key
                stripped_key = key[len(prefix) :]  # Remove prefix
                # Split the remaining key into section and key
                section, key = stripped_key.split("_", 1)

                self.override_value(section, key, value)

    def override_arguments(self, args):
        args = sys.argv[1:] if args is None else args

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


class Config:
    _instance = None

    def __new__(
        cls, configuration_path="configuration.ini", initial_data={}, arguments=None
    ):
        if cls._instance is None:
            cls._instance = SingleConfig(configuration_path, initial_data, arguments)
        return cls._instance


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
