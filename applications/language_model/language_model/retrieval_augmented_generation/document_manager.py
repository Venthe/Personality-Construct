import logging
from typing import TypedDict
from language_model.retrieval_augmented_generation.shared import RetrievedDocument
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector
import psycopg2
import numpy as np
from psycopg2.extensions import register_adapter, AsIs

from _utilities.models import download_model_if_empty
from collections.abc import MutableSequence


class DocumentManagerConnectionKwargs(TypedDict, total=True):
    username: str | None
    password: str | None
    database: str | None
    host: str | None
    port: int | None


class ConnectionKwargs(TypedDict, total=True):
    database_connection: psycopg2.extensions.connection


def new_connection(**kwargs: DocumentManagerConnectionKwargs):
    username = kwargs.get("username", "postgres")
    password = kwargs.get("password", username)
    database = kwargs.get("database", username)
    host = kwargs.get("host", "localhost")
    port = kwargs.get("port", 5432)
    # Establish connection to Postgres database and set up vector extension
    database_connection = psycopg2.connect(
        f"dbname={database} user={username} password={password} host={host} port={port}"
    )
    return database_connection


class DocumentManager:
    _database_connection: psycopg2.extensions.connection

    def __init__(self, **kwargs: ConnectionKwargs | DocumentManagerConnectionKwargs):
        if "database_connection" in kwargs:
            self._database_connection = kwargs["database_connection"]
        else:
            self._database_connection = new_connection(**kwargs)
        self._logger = logging.getLogger(__name__)

        download_model_if_empty(
            "../../resources/models", "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        )
        self._embedding_model = SentenceTransformer(
            model_name_or_path="../../resources/models/sentence-transformers/multi-qa-mpnet-base-dot-v1",
            local_files_only=True,
            device="cpu",
        )  # device="cuda:0"
        self._initialize_table()
        register_vector(self._database_connection)

    def _initialize_table(self, **kwargs):
        with self._database_connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute("DELETE FROM documents WHERE 1=1")
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS document_index on documents USING hnsw (embedding vector_l2_ops)"
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding VECTOR(768),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """
            )
            self._database_connection.commit()

    def add_document(self, document_text):
        embedding = self._embedding_model.encode(document_text).tolist()
        with self._database_connection.cursor() as cursor:
            cursor.execute(
                "INSERT INTO documents (text, embedding) VALUES (%s, %s)",
                (document_text, embedding),
            )
            self._database_connection.commit()
        self._logger.info("Document added successfully.")

    def remove_document(self, document_id):
        # Delete document based on unique ID
        with self._database_connection.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            self._database_connection.commit()
        self._logger.info("Document removed successfully.")

    def retrieve(self, query) -> MutableSequence[RetrievedDocument]:
        query_embedding = self._embedding_model.encode(query)
        results = self._search_in_database(query_embedding)
        return [
            RetrievedDocument(document=result[0], similarity=result[1])
            for result in results
        ]

    def _search_in_database(self, query_embedding):
        with self._database_connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT text,
                       1 - (embedding <=> %s::vector) as similarity
                FROM documents
                ORDER BY similarity DESC
                LIMIT 5
                """,
                (query_embedding,),
            )
            return cursor.fetchall()
