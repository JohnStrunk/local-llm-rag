import hashlib
import os
from typing import Generic, List, TypeVar

import chromadb
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

_chroma_collection_name = "documents"
_chroma_server = os.environ.get("CHROMA_SERVER", "127.0.0.1")
_chroma_port = os.environ.get("CHROMA_PORT", "8000")
_embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME", "BAAI/bge-base-en-v1.5")
_chunk_size = 1000
_chunk_overlap = 100

_T = TypeVar("_T")


def get_embedding_model() -> Embeddings:
    return HuggingFaceBgeEmbeddings(
        model_name=_embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )


# def setup_db() -> VectorStore:
def setup_db() -> Chroma:
    dbclient = chromadb.HttpClient(host=_chroma_server, port=_chroma_port)
    # dbclient.delete_collection(chroma_collection_name)
    collection = dbclient.get_or_create_collection(_chroma_collection_name)
    # print(f"collection has {collection.count()} documents")

    return Chroma(
        client=dbclient,
        embedding_function=get_embedding_model(),
        collection_name=_chroma_collection_name,
    )


def calc_hash(data: bytes | str) -> str:
    """Calculate the hash of some data."""
    if isinstance(data, str):
        data = data.encode()
    return hashlib.sha256(data, usedforsecurity=False).hexdigest()


def hash_file(file_path: str) -> str:
    """Calculate the hash of a file."""
    with open(file_path, "rb") as f:
        data = f.read()
    return calc_hash(data)


# # Filter using VectorStore abstraction
# def filter_documents2(db: VectorStore, texts: List[Document]) -> List[Document]:
#     output = []
#     for doc in texts:
#         result = db.search(
#             query="",
#             search_type="similarity",
#             filter={"hash": doc.metadata["hash"]},
#         )
#         if len(result) == 0:
#             output.append(doc)
#     return output


# Filter using Chroma-specific get() to avoid generating embeddings
def filter_documents(db: Chroma, texts: List[Document]) -> List[Document]:
    """
    Filter out documents that are already in the database.
    """
    # Create a cache of hashes to avoid unnecessary database lookups
    hash_in_db: dict[str, bool] = {}
    lookups = 0
    output = []
    for doc in texts:
        hash_value = doc.metadata["hash"]
        if hash_value in hash_in_db:
            if hash_in_db[hash_value]:
                continue
            else:
                output.append(doc)
                continue
        result = db.get(include=["metadatas"], where={"hash": hash_value})
        lookups += 1
        if len(result["ids"]) == 0:
            hash_in_db[hash_value] = False
            output.append(doc)
        else:
            hash_in_db[hash_value] = True
    print(f"Looked up {lookups} of {len(texts)} hashes in the database")
    return output


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=_chunk_size, chunk_overlap=_chunk_overlap
    )
    texts = []
    for doc in documents:
        txt = text_splitter.split_documents([doc])
        part = 0
        for t in txt:
            # Since we're splitting the documents, we need to make sure each one
            # has a unique ID
            t.metadata["id"] = doc.metadata["id"] + f"_{part}"
            part += 1
        texts.extend(txt)
    return texts


def add_documents_to_db(
    db: VectorStore, documents: List[Document], progress: bool = True
):
    """
    Add documents to the database.
    """
    group_size = 500  # Divide texts into groups so we can save our progress
    groups = [
        documents[i : i + group_size] for i in range(0, len(documents), group_size)
    ]
    done = 0
    for group in groups:
        # Clean up metadata for Chroma
        for doc in group:
            # We're potentially modifying the dict while iterating over it, so we
            # need to make a copy of the items to avoid a RuntimeError
            items = list(doc.metadata.items())
            for key, value in items:
                # Convert lists to strings
                if isinstance(value, list):
                    if len(value) == 0:
                        del doc.metadata[key]
                    else:
                        value = [str(v) for v in value]  # ensure all values are strings
                        doc.metadata[key] = ", ".join(value)
                # Remove None values
                if value is None:
                    del doc.metadata[key]

        ids = [item.metadata["id"] for item in group]
        db.add_documents(group, ids=ids)
        done += len(group)
        if progress:
            print(f"Added {done} of {len(documents)} documents to database")
