import logging
import colorama


class ConsoleFormatter(logging.Formatter):
    _datefmt = "%Y-%m-%dT%H:%M:%S"
    _format = "%(asctime)s.%(msecs)03dZ %(levelname)-5s %(process)d --- [%(name)s][%(threadName)s] : %(message)s (%(filename)s:%(lineno)d)"

    _COLORS = {
        logging.DEBUG: colorama.Fore.LIGHTBLACK_EX,  # Grey
        logging.INFO: colorama.Style.NORMAL,  # Green (to distinguish from DEBUG)
        logging.WARNING: colorama.Fore.YELLOW,  # Yellow
        logging.ERROR: colorama.Fore.RED,  # Red
        logging.CRITICAL: colorama.Fore.RED + colorama.Style.BRIGHT,  # Bold Red
    }
    _reset = colorama.Style.RESET_ALL

    def format(self, record):
        log_color = self.color_for(record.levelno)
        log_fmt = f"{log_color}{self._format}{self._reset}"
        formatter = logging.Formatter(fmt=log_fmt, datefmt=self._datefmt)
        return formatter.format(record)

    def color_for(self, levelno):
        return self._COLORS.get(levelno, self._reset)

    @staticmethod
    def provide_handler(log_level):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(ConsoleFormatter())
        return console_handler


class FileFormatter(logging.Formatter):
    _format = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "process": %(process)d, "name": "%(name)s", "thread": "%(threadName)s", "message": "%(message)s", "file": "%(filename)s", "line": %(lineno)d}'
    _datefmt = "%Y-%m-%dT%H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._format, datefmt=self._datefmt)

    @staticmethod
    def provide_handler(log_level, log_path):
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(FileFormatter())
        return file_handler


def setup_logging(base_log_level=logging.INFO, log_path=None):
    handlers = [ConsoleFormatter.provide_handler(log_level=base_log_level)]

    if log_path is not None:
        handlers.append(
            FileFormatter.provide_handler(log_level=base_log_level, log_path=log_path)
        )

    colorama.init(autoreset=True)
    logging.basicConfig(
        level=base_log_level,
        handlers=handlers,
    )


def map_log_level(log_level_str):
    """
    Map a string representation of a log level to the corresponding logging level.

    Args:
        log_level_str (str): The string representation of the log level (e.g., 'DEBUG', 'info').

    Returns:
        int: The corresponding logging level as an integer.

    Raises:
        ValueError: If the input string does not correspond to a valid log level.
    """
    # Create a mapping of string representations to logging levels
    log_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    # Normalize the input string to upper case
    normalized_level = log_level_str.upper()

    # Retrieve and return the corresponding logging level
    if normalized_level in log_levels:
        return log_levels[normalized_level]
    else:
        raise ValueError(f"Invalid log level: '{log_level_str}'")
