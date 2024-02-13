#!/usr/bin/env python
import datetime
import os
from calendar import c

from langchain_community.document_loaders import NotionDBLoader

from common import (
    add_documents_to_db,
    calc_hash,
    filter_documents,
    setup_db,
    split_documents,
)

token = os.environ.get("NOTION_TOKEN", "")
dbid = os.environ.get("NOTION_DB_ID", "")


def main():
    print(f"{datetime.datetime.now()} - Loading documents from Notion database...")
    loader = NotionDBLoader(
        integration_token=token,
        database_id=dbid,
        request_timeout_sec=30,
    )
    raw_docs = loader.load()
    if len(raw_docs) == 0:
        print("No documents found")
        return
    print(f"Loaded {len(raw_docs)} documents")

    # Add standard metadata to the documents
    for doc in raw_docs:
        doc.metadata["hash"] = calc_hash(doc.page_content)
        doc.metadata["loader"] = "NotionDBLoader"
        # Try to get a descriptive title for the source of the document
        source = "Notion - " + doc.metadata["id"]
        source = doc.metadata.get("title", source)
        source = doc.metadata.get("name", source)
        doc.metadata["source"] = source

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
