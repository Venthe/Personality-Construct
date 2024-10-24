from huggingface_hub import snapshot_download
import os
import logging

def download_model_if_empty(model_path, model_name):
    logger = logging.getLogger(__name__)
    full_path = os.path.join(model_path, model_name)
    
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    elif os.listdir(full_path):
        logger.debug(f"The directory '{full_path}' is not empty.")
        return
    
    snapshot_download(local_dir=full_path, repo_id=model_name)