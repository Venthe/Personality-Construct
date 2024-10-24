# TODO: Implement modular RAG
#  While I acknowledge that the Modular RAG approach seems best,
#  I have too little experience to understand the flow between elements
#  Essentially, how to map the whitepaper to the code.
# See also:
# - https://arxiv.org/html/2407.21059v1
# - https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-%E2%85%B0-e69b32dc13a3
# - https://medium.com/@yufan1602/modular-rag-and-rag-flow-part-ii-77b62bf8a5d3
# - https://medium.com/@sahin.samia/modular-rag-using-llms-what-is-it-and-how-does-it-work-d482ebb3d372
# - https://adasci.org/how-does-modular-rag-improve-upon-naive-rag/
from typing import MutableSequence, Union
from language_model.language_model.language_model import LanguageModel
from language_model.language_model.mistral_chat_template import assistant_message
from language_model.retrieval_augmented_generation.document_manager import (
    DocumentManager,
)
from language_model.retrieval_augmented_generation.refiner import Refiner
from language_model.retrieval_augmented_generation.reranker import (
    Reranker,
    filter_relevant,
)
import logging


class SequentialRAGPipeline:
    def __init__(
        self,
        language_model: LanguageModel,
        retriever: DocumentManager,
        refiner: Refiner,
        reranker: Reranker,
    ):
        self._language_model = language_model
        self._retriever = retriever
        self._refiner = refiner
        self._reranker = reranker
        self._logger = logging.getLogger(__name__)

    def execute(
        self,
        prompt,
        token_generated_callback=None,
        response_generated_callback=None,
    ) -> str:

        query = prompt[-1]["content"]

        self._logger.debug(f"Retrieving relevant documents for: {query}")
        documents = self._retriever.retrieve(query)
        self._logger.debug(f"Retrievied documents: {len(documents)}")
        if len(documents) > 0:
            self._logger.debug(f"Reranking documents")
            reranked_documents = filter_relevant(
                self._reranker.rerank(query, documents)
            )
            self._logger.debug(f"Reranked documents: {len(reranked_documents)}")
            if len(reranked_documents) > 0:
                text_to_add_to_assistant = self._refiner.refine(reranked_documents)
                self._logger.debug(
                    f"Prerefined information: {[d.document for d in reranked_documents]}\nRefined information: {text_to_add_to_assistant}"
                )
                prompt.append(
                    assistant_message(
                        content=f"Knowing the information from the documents:\n{text_to_add_to_assistant}\n\nmy response, even if I thought previously otherwise, based on that is:",
                        eos=False,
                    )
                )
        self._logger.debug(f"Generating request with:\n{prompt}")
        return self._language_model.generate(
            prompt,
            token_generated_callback=token_generated_callback,
            response_generated_callback=response_generated_callback,
        )
