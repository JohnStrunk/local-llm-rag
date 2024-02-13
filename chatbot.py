#!/usr/bin/env python3
# Initially from https://github.com/ollama/ollama/blob/main/examples/langchain-python-rag-privategpt/privateGPT.py

import argparse
import os
import time

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama

from common import setup_db

model = os.environ.get("LLM_MODEL_NAME", "mistral:7b-instruct-q5_K_M")
target_source_chunks = int(os.environ.get("TARGET_SOURCE_CHUNKS", 4))


def main():
    # Parse the command line arguments
    args = parse_arguments()

    db = setup_db()

    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = []  # if args.mute_stream else [StreamingStdOutCallbackHandler()]

    llm = Ollama(model=model, callbacks=callbacks)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=not args.hide_source,
    )
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query (or 'exit'): ")
        if query.lower() == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        # XXX: Switched from __call__ to invoke accd to deprecation warning, but
        # invoke wants a dict, not a string. TODO: Figure out what the dict
        # should be.
        res = qa.invoke(query)
        answer, docs = res["result"], (
            [] if args.hide_source else res["source_documents"]
        )
        end = time.time()

        print(f"\nAnswer ({(end-start):.2f}):\n{answer}")

        # Print the relevant sources used for the answer
        print("\nSources:")
        # Filter docs to only be unique items
        unique_docs = list({doc.metadata["source"]: doc for doc in docs}.values())
        for document in unique_docs:
            print("> " + document.metadata["source"])
            # print(document.page_content)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="privateGPT: Ask questions to your documents without an internet connection, "
        "using the power of LLMs."
    )
    parser.add_argument(
        "--hide-source",
        "-S",
        action="store_true",
        help="Use this flag to disable printing of source documents used for answers.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
