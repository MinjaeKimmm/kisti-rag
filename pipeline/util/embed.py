import pandas as pd
from tqdm import tqdm
import chromadb

from langchain_chroma import Chroma

from ..common import embedding_function, embedding_dir
from .kisti_data import get_sample_paper

"""Splits a list into a list of lists, each with length no more than chunk_size.

Read below link for why this is needed when adding embeddings to collection.
https://github.com/chroma-core/chroma/issues/1049#issuecomment-1699859480
"""
def split_list(input_list, chunk_size):
    res = []
    for i in range(0, len(input_list), chunk_size):
        res.append(input_list[i:i + chunk_size])
    return res

"""Create collection of embeddings.

name: name of created collection
splitter: function that splits each document's contents into chunks
"""
def embed(name, splitter, with_metadata=False):
    client = chromadb.PersistentClient(embedding_dir)
    print(name)
    db = Chroma(client=client, collection_name=name, embedding_function=embedding_function)
    if db._collection.count() > 0:
        print(f'Found {db._collection.count()} embeddings')
        return
    data = get_sample_paper()
    documents = []
    metadatas = []
    skipped_docs = 0
    print('Processing data...')
    for doc in tqdm(data):
        if not doc.page_content.strip():
            skipped_docs += 1
            continue
        texts_splitted = splitter(doc.page_content)
        for text in texts_splitted:
            if with_metadata:
                text = f'content: {text}'
                if 'keywords' in doc.metadata:
                    text = f'keywords: {doc.metadata["keywords"]} {text}'
            documents.append(text)
            metadatas.append(doc.metadata)
    print(f'Skipped {skipped_docs} empty or unsplittable documents')
    if not documents:
        print("Error: No valid documents to embed")
        return
    ids = list(map(str, range(len(documents))))
    print(f'Received {len(documents)} chunks')
    print(f'Example document: {documents[0]}')
    print(f'Example metadata: {metadatas[0]}')
    print('Embedding...')
    
    batch_size = 1000
    for i in tqdm(range(0,len(documents), batch_size)):
        batch_end = min(i + batch_size, len(documents))
        batch_documents = documents[i:batch_end]
        batch_metadatas = metadatas[i:batch_end]
        batch_ids = ids[i:batch_end]

        batch_embeddings = embedding_function.embed_documents(batch_documents)

        db._collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            documents=batch_documents
        )
    if with_metadata:
        remove_small_chunks(name, 50)
    else:
        remove_small_chunks(name)

"""Deletes chroma collection of the name."""
def delete_embeddings(name):
    client = chromadb.PersistentClient(embedding_dir)
    client.delete_collection(name)
    print(f'Deleted collection {name}')

"""Removes chunks of length below threshold in collection of given name.

Chunks of size too small to be meaningful will lower retrieval accuracy.
Always run this after embedding.
"""
def remove_small_chunks(name, threshold=20):
    print(f'Removing small chunks for {name}...')
    db = Chroma(name, embedding_function, embedding_dir)
    results = db._collection.get()
    remove_ids = []
    for i in tqdm(range(len(results['ids']))):
        if len(results['documents'][i]) < threshold:
            remove_ids.append(results['ids'][i])
    if len(remove_ids) > 0:
        remove_ids_chunked = split_list(remove_ids, 41000)
        for ids in remove_ids_chunked:
            db._collection.delete(ids)
        print('Removed.')
    else:
        print('Already Removed.')

if __name__ == '__main__':
    client = chromadb.PersistentClient(embedding_dir)
    print(client.list_collections())
