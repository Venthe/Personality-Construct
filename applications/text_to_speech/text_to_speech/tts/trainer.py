import os
from openvoice import se_extractor
import shutil
import logging


class Trainer:
    def __init__(self, create_tone_converter_callback):
        self.__logger = logging.getLogger(__name__)
        self.__tone_converter = create_tone_converter_callback()

    def train(
        self,
        reference_file,
        target_directory="./output/",
        use_vad=True,
        name="se",
        clean=True,
    ):
        self.__logger.info(f"Generating embedding to {target_directory} from {reference_file}")
        target_se, audio_name = se_extractor.get_se(
            audio_path=reference_file,
            vc_model=self.__tone_converter,
            vad=use_vad,
            target_dir=target_directory,
        )
        self.__logger.debug(f"Embedding generated {target_directory}")

        generated_directory = os.path.join(target_directory, audio_name)
        target_directory = os.path.join(target_directory, name)
        self.__logger.debug(f"Directories. {generated_directory} {target_directory}")
        if clean:
            self.__logger.debug(f"clean: {clean}")
            shutil.rmtree(target_directory, ignore_errors=True)
        os.makedirs(target_directory, exist_ok=True)

        for item in os.listdir(generated_directory):
            self.__logger.debug(f"Item: {item}")
            if item == "wavs":
                continue
            existing_item_path = os.path.join(generated_directory, item)
            target_item_path = os.path.join(target_directory, item)

            if item.endswith(".pth"):
                filename = "checkpoint.pth"
                target_item_path = os.path.join(target_directory, filename)
            self.__logger.debug(f"Moving {existing_item_path} {target_item_path}")
            shutil.move(existing_item_path, target_item_path)

        shutil.rmtree(generated_directory, ignore_errors=True)
        self.__logger.info(f"Embeddigng generated to {target_directory}")
