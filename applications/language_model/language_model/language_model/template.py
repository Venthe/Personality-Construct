import os


def get_chat_template():
    file_path = os.path.join(
        "resources", "mistral_nemo_chat_template-modified.j2"
    )
    with open(file_path, "r") as file:
        chat_template = file.read()
    return chat_template
