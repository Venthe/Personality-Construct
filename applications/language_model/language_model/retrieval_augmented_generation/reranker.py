from typing import MutableSequence
from language_model.retrieval_augmented_generation.shared import RetrievedDocument
from sentence_transformers import CrossEncoder
from _utilities.models import download_model_if_empty


class Reranker:
    def __init__(
        self,
        model_path="models",
        model_name="cross-encoder/ms-marco-MiniLM-L-2-v2",
        use_gpu=False,
    ):
        download_model_if_empty(
            "../../resources/models", "cross-encoder/ms-marco-MiniLM-L-2-v2"
        )
        self._model = CrossEncoder(
            "../../resources/models/cross-encoder/ms-marco-MiniLM-L-2-v2",
            max_length=512,
            device="cpu",
        )

    def rerank(self, query, *args: MutableSequence[RetrievedDocument]):
        pairs = self._create_pairs(query, *args)
        return _zip_scores(*args, self._model.predict(pairs))

    def _create_pairs(self, query, args):
        return [(query, arg.document) for arg in args]


def _zip_scores(
    documents: MutableSequence[RetrievedDocument], scores
) -> MutableSequence[RetrievedDocument]:
    return sorted(
        [
            RetrievedDocument(doc.document, score)
            for doc, score in zip(documents, scores)
        ],
        key=lambda x: x.similarity,
        reverse=True,
    )

def filter_relevant(
    documents: MutableSequence[RetrievedDocument]
) -> MutableSequence[RetrievedDocument]:
    return [value for value in documents if value.similarity > 0 ]