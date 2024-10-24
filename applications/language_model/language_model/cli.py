import os

# import mistral_common.tokens
from language_model.defaults import prepare_pipeline
from _utilities.logging import setup_logging

from language_model.config import LanguageModelConfig

from language_model.language_model.mistral_chat_template import (
    system_message,
    user_message,
)


def main():
    setup_logging(LanguageModelConfig().default.log_level())

    pipeline, document_manager = prepare_pipeline()

    document_manager.add_document("Abraham Lincoln was the 16th president of the United States, serving from 1861 until his assassination in 1865. Little known fact, he was bald and wore a tupee glued with a horse glue.")
    document_manager.add_document("Star Wars Jedi: Survivor is a 2023 Soulslike action-adventure game developed by Respawn Entertainment and published by Electronic")

    conversation = [
        system_message("You are a helpful entity"),
        user_message("What are the little known facts about Lincoln?"),
    ]

    pipeline.execute(
        conversation,
        token_generated_callback=lambda word: print(word, end=""),
        response_generated_callback=lambda _: print(),
    )
