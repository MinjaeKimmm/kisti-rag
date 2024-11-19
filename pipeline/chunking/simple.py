from tqdm import tqdm
import chromadb

import os
import pickle

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from ..util.embed import embed
from ..common import embedding_function, embedding_dir, kiwi_tokenizer
from ..util.kisti_data import get_sample_paper

"""Returns name for chroma collection for embeddings"""
def get_simple_collection_name(chunk_size, chunk_overlap, with_metadata=False):
    text = f'kisti-2-{chunk_size}-{chunk_overlap}'
    if with_metadata:
        text = f'{text}-metadata'
    return text

"""Returns function that executes simple chunking."""
def get_simple_splitter(chunk_size, chunk_overlap):
    def splitter(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return text_splitter.split_text(text)
    return splitter

"""Create the collection of embeddings."""
def embed_simple(chunk_size, chunk_overlap, with_metadata=False):
    name = get_simple_collection_name(chunk_size, chunk_overlap, with_metadata)
    splitter = get_simple_splitter(chunk_size, chunk_overlap)
    embed(name, splitter, with_metadata)

"""Returns langchain retriever with simple chunking method and embedding.

Simple chunking:
Divides document into chunks of size and overlap of about chunk_size and chunk_overlap, measured in string length.
"""
def get_simple_retriever(type, chunk_size, chunk_overlap, with_metadata=False):
    embed_simple(chunk_size, chunk_overlap, with_metadata)
    client = chromadb.PersistentClient(embedding_dir)
    name = get_simple_collection_name(chunk_size, chunk_overlap, with_metadata)
    db = Chroma(client=client, collection_name =name, embedding_function=embedding_function)
    if db._collection.count() == 0:
        print('Embeddings not found')
        return
    if (type == 'dense'):
        return db.as_retriever()
    elif (type == 'bm25'):
        cache_dir = os.path.join(embedding_dir, 'bm25_cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_file_name = f'simple_2_{chunk_size}_{chunk_overlap}'
        if with_metadata:
            cache_file_name = f'{cache_file_name}_metadata'
        cache_file = os.path.join(cache_dir, f'{cache_file_name}.pkl')

        if os.path.exists(cache_file):
            print('Loading cached BM25 Retriever')
            with open(cache_file, 'rb') as f:
                retriever = pickle.load(f)
                return retriever

        print('Creating new BM25 Retriever')
        documents = db.get()
        bm25_retriever = BM25Retriever.from_texts(tqdm(documents['documents']), preprocess_func=kiwi_tokenizer)

        print('Saving BM25 Retriever to cache')
        with open(cache_file, 'wb') as f:
            pickle.dump(bm25_retriever, f)

        return bm25_retriever
    else:
        raise ValueError(f"Unknown retriever type: {type}")

if __name__ == '__main__':
    client = chromadb.PersistentClient(embedding_dir)
    name = get_simple_collection_name(500, 50)
    db = Chroma(client=client, collection_name=name, embedding_function=embedding_function)
    documents = db.get()
    print(documents.keys())
    print(documents['ids'][0])
    print(documents['metadatas'][0])
    print(documents['documents'][0])
