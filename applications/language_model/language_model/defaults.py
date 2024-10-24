import os
from language_model.config import LanguageModelConfig
from language_model.language_model.language_model import LanguageModel
from language_model.retrieval_augmented_generation.document_manager import (
    DocumentManager,
)
from language_model.retrieval_augmented_generation.pipeline import SequentialRAGPipeline
from language_model.retrieval_augmented_generation.refiner import Refiner
from language_model.retrieval_augmented_generation.reranker import Reranker


def prepare_pipeline():
    dm = DocumentManager(username="postgres")
    reranker = Reranker()
    model = LanguageModel(
        f"{os.path.join(LanguageModelConfig().llm.model_path(), LanguageModelConfig().llm.model_name())}",
        device_map="cuda:0",
    )
    refiner = Refiner(model)
    return (
        SequentialRAGPipeline(
            language_model=model, refiner=refiner, reranker=reranker, retriever=dm
        ),
        dm,
    )
