#!/usr/bin/env python
import datetime
import glob
import os
from multiprocessing import Pool
from typing import List

from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.vectorstores import VectorStore

from common import (
    add_documents_to_db,
    filter_documents,
    hash_file,
    setup_db,
    split_documents,
)

# Load environment variables
source_directory = os.environ.get("SOURCE_DIRECTORY", "source_documents")

# # Custom document loaders
# class MyElmLoader(UnstructuredEmailLoader):
#     """Wrapper to fallback to text/plain when default does not work"""

#     def load(self) -> List[Document]:
#         """Wrapper adding fallback for elm without html"""
#         try:
#             try:
#                 doc = UnstructuredEmailLoader.load(self)
#             except ValueError as e:
#                 if "text/html content not found in email" in str(e):
#                     # Try plain text
#                     self.unstructured_kwargs["content_source"] = "text/plain"
#                     doc = UnstructuredEmailLoader.load(self)
#                 else:
#                     raise
#         except Exception as e:
#             # Add file_path to exception message
#             raise type(e)(f"{self.file_path}: {e}") from e

#         return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {"mode": "elements"}),
    # ".docx": (Docx2txtLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {"mode": "elements"}),
    ".enex": (EverNoteLoader, {}),
    # ".eml": (MyElmLoader, {}),
    # ".epub": (UnstructuredEPubLoader, {"mode": "elements"}),
    ".html": (UnstructuredHTMLLoader, {"mode": "elements"}),
    ".md": (UnstructuredMarkdownLoader, {"mode": "elements"}),
    ".odt": (UnstructuredODTLoader, {"mode": "elements"}),
    ".pdf": (PyMuPDFLoader, {"sort": True}),
    ".ppt": (UnstructuredPowerPointLoader, {"mode": "elements"}),
    ".pptx": (UnstructuredPowerPointLoader, {"mode": "elements"}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        try:
            loader = loader_class(file_path, **loader_args)
            docs = loader.load()
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
        part = 0
        for doc in docs:
            doc.metadata["id"] = file_path.replace(source_directory, "", 1) + f"_{part}"
            doc.metadata["loader"] = loader_class.__name__
            doc.metadata["source"] = file_path.replace(source_directory, "", 1)
            doc.metadata["hash"] = hash_file(file_path)
            part += 1
        return docs
    return []


def load_documents(source_dir: str) -> List[Document]:
    """
    Loads all documents from the source documents directory
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )

    # Load docs in parallel
    process_count = os.cpu_count() or 1
    with Pool(processes=process_count) as pool:
        results = []
        for _, docs in enumerate(pool.imap_unordered(load_single_document, all_files)):
            results.extend(docs)

    return results


def main():
    print(f"{datetime.datetime.now()} - Loading documents from source directory...")
    raw_docs = load_documents(source_directory)
    if len(raw_docs) == 0:
        print("No documents found in source directory")
        return
    print(f"Loaded {len(raw_docs)} document fragments from source directory")

    print(f"{datetime.datetime.now()} - Filtering out already processed documents...")
    db = setup_db()
    filtered_docs = filter_documents(db, raw_docs)
    if len(filtered_docs) == 0:
        print("No new or updated documents to add to database")
        return
    print(
        f"Filtered out {len(raw_docs) - len(filtered_docs)} already processed documents"
    )

    print(f"{datetime.datetime.now()} - Splitting documents into chunks...")
    chunks = split_documents(filtered_docs)
    if len(chunks) == 0:
        print("No new chunks to add to database")
        return

    print(
        f"{datetime.datetime.now()} - Creating embeddings for {len(chunks)} chunks..."
    )
    add_documents_to_db(db, chunks, progress=True)

    print(f"{datetime.datetime.now()} - Done!")


if __name__ == "__main__":
    main()
