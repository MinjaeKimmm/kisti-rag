import random

# Langchain imports
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Local imports
from ..common import duplicate, remove_metadata

def remove_duplicate_docs(docs):
    """Returns list of Documents without the duplications. Order preserved."""
    results = []
    for doc in docs:
        if any(map(lambda d:duplicate(d, doc), results)): continue
        results.append(doc)
    return results

def set_retriever_k(retriever, k):
    if isinstance(retriever, BM25Retriever):
        retriever.__dict__['k'] = k
        retriever.__dict__['_k'] = k  
    else:
        retriever.search_kwargs['k'] = k

def get_k_from_plain_retriever(retriever, k, query):
    docs = []
    k_arg = 2*k
    while len(docs) < k:
        set_retriever_k(retriever, k_arg)
        docs = retriever.invoke(query)
        docs = remove_duplicate_docs(docs)
        k_arg *= 2
    return docs[:k]

def get_k_from_ensemble_retriever(retriever, k, query):
    for i in range(len(retriever.retrievers)):
        set_retriever_k(retriever.retrievers[i], k)
    docs = retriever.invoke(query)
    docs = remove_duplicate_docs(docs)
    return docs[:k]

def get_k_from_retriever(retriever, k, query):
    """Returns top k documents, without duplications.
    Always use this instead of .invoke method."""
    if isinstance(retriever, EnsembleRetriever):
        return get_k_from_ensemble_retriever(retriever, k, query)
    return get_k_from_plain_retriever(retriever, k, query)

