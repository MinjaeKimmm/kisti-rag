import os
import shutil
import warnings
from tqdm import tqdm
import json

from langchain._api import LangChainDeprecationWarning
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain_core.retrievers import BaseRetriever, RetrieverLike
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from ..util.kisti_data import get_sample_paper
from ..common import embedding_function, embedding_dir
from ..util.embed import remove_small_chunks, delete_embeddings

"""Returns name for chroma collection for child chunk embeddings"""
def get_sentence_child_collection_name(parent_chunk_size, child_chunk_size):
    return f'kisti-sentence-child-2-{parent_chunk_size}-{child_chunk_size}'
"""Returns name for folder for parent documents"""
def get_sentence_parent_folder_name(parent_chunk_size, child_chunk_size):
    return f'kisti-sentence-parent-2-{parent_chunk_size}-{child_chunk_size}'

"""Returns langchain retriever with small-to-big method and embedding, splitting by sentences.
View chunking_parent.py file for more information.
"""
def get_sentence_parent_retriever(parent_chunk_size, child_chunk_size):
    child_name = get_sentence_child_collection_name(parent_chunk_size, child_chunk_size)
    warnings.filterwarnings('ignore', category=LangChainDeprecationWarning)
    vectorstore = Chroma(collection_name=child_name, embedding_function=embedding_function, persist_directory=embedding_dir)
    # https://stackoverflow.com/questions/77385587/persist-parentdocumentretriever-of-langchain
    fs = LocalFileStore(os.path.join(embedding_dir, get_sentence_parent_folder_name(parent_chunk_size, child_chunk_size)))
    store = create_kv_docstore(fs)
    #child_splitter = CharacterTextSplitter('.', chunk_size=child_chunk_size, chunk_overlap=0, add_start_index=True)
    child_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", ".", "\n"], chunk_size=child_chunk_size, chunk_overlap=0, add_start_index=True)
    #parent_splitter = CharacterTextSplitter('.', chunk_size=parent_chunk_size, chunk_overlap=0, add_start_index=True)
    parent_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", ".", "\n"], chunk_size=parent_chunk_size, chunk_overlap=0, add_start_index=True)
    retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter, parent_splitter=parent_splitter)
    # Checks if file store is empty
    if next(fs.yield_keys(), None) is None:
        docs = get_sample_paper()
        print('Adding Documents...')
        for doc in tqdm(docs):
            # Sometimes an error is caused by empty texts
            # See 의료법학/2008/미용성형수술의 특수성 for example
            try:
                retriever.add_documents([doc])
            except:
                pass
        remove_small_chunks(child_name)
    return retriever

"""Deletes all data on disk related to the parent retriever."""
def delete_sentence_parent_retriever_data(parent_chunk_size, child_chunk_size):
    delete_embeddings(get_sentence_child_collection_name(parent_chunk_size, child_chunk_size))
    parent_path = os.path.join(embedding_dir, get_sentence_parent_folder_name(parent_chunk_size, child_chunk_size))
    shutil.rmtree(parent_path)

def check_stored_parent_chunks(parent_chunk_size, child_chunk_size):
    # Get the parent folder name
    parent_folder_name = get_sentence_parent_folder_name(parent_chunk_size, child_chunk_size)
    parent_path = os.path.join(embedding_dir, parent_folder_name)
    
    # Create LocalFileStore and docstore
    fs = LocalFileStore(parent_path)
    store = create_kv_docstore(fs)
    
    chunk_lengths = []
    
    # Iterate through all keys in the store
    for key in fs.yield_keys():
        # Get the document from the store
        doc = store.mget([key])[0]
        if doc and isinstance(doc, Document):
            # Add the length of the document's content to our list
            chunk_lengths.append(len(doc.page_content))
    
    if chunk_lengths:
        print(f"Parent chunk lengths:")
        print(f"Min: {min(chunk_lengths)}")
        print(f"Max: {max(chunk_lengths)}")
        print(f"Average: {sum(chunk_lengths) / len(chunk_lengths):.2f}")
        print(f"Total chunks: {len(chunk_lengths)}")
    else:
        print("No parent chunks found in storage.")


if __name__ == '__main__':
    check_stored_parent_chunks(500,125)
    check_stored_parent_chunks(501,125)
    check_stored_parent_chunks(502,125)