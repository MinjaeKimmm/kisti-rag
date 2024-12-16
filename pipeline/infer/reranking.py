import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ..common import remove_metadata
from ..llm import format_prompt, generate
from .retriever import get_k_from_retriever

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def doc_reranker(query, docs):
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        model = 'BAAI/bge-reranker-base'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
        inputs = tokenizer(pairs, padding='max_length', truncation=False, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
    sorted_pairs = [pair for _, pair in sorted(zip(scores, pairs), reverse=True, key=lambda x: x[0])]
    return sorted_pairs

def get_retrieval_chain_w_rerank(retriever, k):
    def retrieval_chain_w_rerank(query):
        docs = get_k_from_retriever(retriever, k, query)
        docs_reranked = doc_reranker(query, docs)
        context = '\n'.join([remove_metadata(doc[1]) for doc in docs_reranked])
        formatted_prompt = format_prompt(query, context)
        output = generate(formatted_prompt)
        return output
    return retrieval_chain_w_rerank
