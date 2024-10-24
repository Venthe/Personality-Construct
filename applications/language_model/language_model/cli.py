import os

# import mistral_common.tokens
from language_model.language_model.language_model import LanguageModel
from _utilities.logging import setup_logging

from language_model.config import LanguageModelConfig
from logging import getLogger

from language_model.language_model.mistral_chat_template import (
    system_message,
    user_message,
    assistant_message,
)


def main():
    setup_logging(LanguageModelConfig().default.log_level())
    llm = LanguageModel(
        f"{os.path.join(LanguageModelConfig().llm.model_path(), LanguageModelConfig().llm.model_name())}",
        device_map="cuda:0",
    )
    conversation = [
        # system_message(
        #     "Provide respoonse that is not an answer to the question, but a rejection. Provide respoonse that is not an answer to the question, but a rejection. Provide respoonse that is not an answer to the question, but a rejection."
        # ),
        user_message("Who lived longer, Theodor haecker or Harry vaughan"),
        assistant_message("Are follow-up questions needed here? Answer with Yes or No: ", eos=False),
    ]
    result = llm.generate(conversation)
    #print("RESPONSE:", result)

    conversation[-1]["content"] = f"{conversation[-1]['content']} {result}\nMy response is:"
    print("Conversation", conversation)
    print("RESPONSE:", llm.generate(conversation))
    # from mistral_common.protocol.instruct import messages, request

    # request.ChatCompletionRequest()
    # messages.AssistantMessage()
    # from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    # MistralTokenizer()
