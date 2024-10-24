import os
import logging
from openvoice.api import ToneColorConverter


def create_tone_converter(converter_path, device="cpu"):
    logger = logging.getLogger(__name__)
    logger.debug(f"Creating tone converter for path {converter_path}")
    config_path = os.path.normpath(os.path.join(converter_path, "config.json"))
    model_path = os.path.normpath(os.path.join(converter_path, "checkpoint.pth"))
    tone_converter = ToneColorConverter(config_path, device=device)
    tone_converter.load_ckpt(model_path)

    logger.info(f"Tone converter created for path {converter_path}")
    return tone_converter
