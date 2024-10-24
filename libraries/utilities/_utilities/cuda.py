import logging


def detect_cuda():
    try:
        import torch

        logger = logging.getLogger(__name__)
        if torch == None:
            return False

        if torch.cuda.is_available():
            logger.debug("CUDA is available.")
            device_mapping = {
                i: torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            }
            device_mapping_output = "\n".join(
                f"Index: {index}, Device Name: {name}"
                for index, name in device_mapping.items()
            )
            logger.debug(f"CUDA devices: {device_mapping_output}")
            return True
        else:
            logger.debug("CUDA is not available.")
            return False
    except ImportError:
        logger.warning("Torch is not installed")
        return False
