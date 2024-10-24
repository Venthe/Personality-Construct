import os
from openvoice import se_extractor
import shutil
import logging
from openvoice.api import ToneColorConverter
from text_to_speech.tone_converter.tone_converter import ToneColorConverterWrapper


class ToneConverterTrainer:
    def __init__(self, tone_color_converter: ToneColorConverterWrapper):
        self._logger = logging.getLogger(__name__)
        self._tone_converter = tone_color_converter

    def train(
        self,
        reference_file,
        target_directory="./output/",
        use_vad=True,
        name="checkpoint",
        clean=True,
    ):
        self._logger.info(
            f"Generating embedding to {target_directory} from {reference_file}"
        )
        target_se, audio_name = se_extractor.get_se(
            audio_path=reference_file,
            vc_model=self._tone_converter.model,
            vad=use_vad,
            target_dir=target_directory,
        )
        self._logger.debug(f"Embedding generated {target_directory}")

        generated_directory = os.path.join(target_directory, audio_name)
        target_directory = os.path.join(target_directory, name)
        self._logger.debug(f"Directories. {generated_directory} {target_directory}")
        if clean:
            self._logger.debug(f"clean: {clean}")
            shutil.rmtree(target_directory, ignore_errors=True)
        os.makedirs(target_directory, exist_ok=True)

        for item in os.listdir(generated_directory):
            self._logger.debug(f"Item: {item}")
            if item == "wavs":
                continue
            existing_item_path = os.path.join(generated_directory, item)
            target_item_path = os.path.join(target_directory, item)

            if item.endswith(".pth"):
                filename = "checkpoint.pth"
                target_item_path = os.path.join(target_directory, filename)
            self._logger.debug(f"Moving {existing_item_path} {target_item_path}")
            shutil.move(existing_item_path, target_item_path)

        shutil.rmtree(generated_directory, ignore_errors=True)
        self._logger.info(f"Embedding generated to {target_directory}")
