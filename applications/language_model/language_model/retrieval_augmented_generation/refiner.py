from typing import MutableSequence
from language_model.language_model.language_model import LanguageModel
from language_model.language_model.mistral_chat_template import system_message, user_message
from language_model.retrieval_augmented_generation.shared import RetrievedDocument


class Refiner:
    def __init__(self, model: LanguageModel):
        self._model = model

    def refine(self, documents: MutableSequence[RetrievedDocument]) -> str:
        documents_for_summarization = "\n".join([
            f"{idx + 1}. {value.document}" for idx, value in enumerate(documents)
        ])
        messages = [
            system_message(
                """
                You are a perfect desk clerk incapable of making mistakes.
                You respond diligently and without missing any information.
                You do not create new information.
                You never lie.
                You never cheat.
                You are writing a document, so you should always write in a matter-of-fact manner.
                You are always responding as concrete as possible.
                You are writing without additional, useless adornments.
                You do not write headings.
                You describe the documents you are provided.
                When asked for details, you write every single one, adding relevancy note.
                You summarize the following document or documents while preserving the details in the following format:
                - Fragment: <fragment verbatim>, Summary: <fragment summary>, Details: <fragment details>
            """
            ),
            user_message(
                f"Summarize the following document or documents while preserving the the details:\n{documents_for_summarization}"
            ),
        ]
        return self._model.generate(messages)
