from .utilities import resample_audio, to_soundfile
import torch
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
        self.__logger.info(
            f"Generating embedding to {target_directory} from {reference_file}"
        )
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


class Embedder:
    def __init__(
        self,
        create_tone_converter_callback,
        speaker_model,
        embedding_model,
        device="cpu",
    ):
        self.__device = device
        self.__tone_converter = create_tone_converter_callback()
        self.__tone_converter_sampling_rate = (
            self.__tone_converter.hps.data.sampling_rate
        )
        self.__tone_convert = self.__init_tone_convert(speaker_model, embedding_model)

    def __init_tone_convert(self, speaker_model, embedding_model):
        speaker_model = torch.load(
            speaker_model, map_location=torch.device(self.__device)
        )
        embedding_model = torch.load(
            embedding_model, map_location=torch.device(self.__device)
        )

        def tone_convert(audio, tau=0.2):
            return self.__tone_converter.convert(
                audio_src_path=audio,
                src_se=speaker_model,
                tgt_se=embedding_model,
                tau=tau,
            )

        return tone_convert

    def embedding(self, tau=0.2):
        def apply(result, sampling_rate):
            result, _ = resample_audio(result, self.__tone_converter_sampling_rate)
            tc = self.__tone_convert(result, tau=tau)
            result, sampling_rate = to_soundfile(
                tc, self.__tone_converter_sampling_rate
            )
            return result, sampling_rate

        return apply
