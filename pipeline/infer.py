# Standard library imports
import math
import json
import random
import re

# Third-party imports
import torch
import numpy as np
from tqdm import tqdm
from nltk import ngrams
from konlpy.tag import Okt
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Langchain imports
from langchain.retrievers.parent_document_retriever import ParentDocumentRetriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# Local imports
from reference.eval.eval import input_file_path, output_file_path, evaluate_by_dicts
from reference.eval.llama3_eval import llm_output_file_path, RAG_eval_w_LLM
from .common import preprocess_text, duplicate, remove_metadata, text_wrap
from .kisti_data import get_sample_qa
from .llm import format_prompt, generate, format_hyde_prompt, hyde_generate

# Load test data
data = get_sample_qa()

# Scoring and normalization functions
def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

# Reranking functions
def doc_reranker(query, docs):
    pairs = [[query, doc.page_content] for doc in docs]
    with torch.no_grad():
        model = 'Dongjin-kr/ko-reranker'
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSequenceClassification.from_pretrained(model)
        inputs = tokenizer(pairs, padding='max_length', truncation=False, return_tensors='pt', max_length=512)
        scores = model(**inputs, return_dict=True).logits.view(-1, ).float()
        scores = exp_normalize(scores.numpy())
    sorted_pairs = [pair for _, pair in sorted(zip(scores, pairs), reverse=True, key=lambda x: x[0])]
    return sorted_pairs

# Retrieval chain functions
def get_retrieval_chain_w_rerank(retriever, k):
    def retrieval_chain_w_rerank(query):
        docs = get_k_from_retriever(retriever, k, query)
        docs_reranked = doc_reranker(query, docs)
        context = '\n'.join([remove_metadata(doc[1]) for doc in docs_reranked])
        formatted_prompt = format_prompt(query, context)
        output = generate(formatted_prompt)
        return output
    return retrieval_chain_w_rerank

def get_retrieval_chain(retriever, k, shuffle=False):
    """Returns function that acts as retrieval chain with given retriever
    Note that it is not the retrieval chain in the Langchain framework, but just a function with string input and output.
    """
    def retrieval_chain(query):
        docs = get_k_from_retriever(retriever, k, query)
        if shuffle:
            random.shuffle(docs)
        context = '\n'.join([remove_metadata(doc.page_content) for doc in docs])
        formatted_prompt = format_prompt(query, context)
        output = generate(formatted_prompt)
        return output
    return retrieval_chain

# HyDE (Hypothetical Document Embeddings) functions
def hyde_query_generate(query, hyde, hyde_logger=None):
    if not hyde:
        return query

    formatted_prompt = format_hyde_prompt(query, chunk_size=500)
    hyde_output = hyde_generate(formatted_prompt, chunk_size=500)
    hyde_query = (query + '\n') * 4 + hyde_output[:125]
    
    if hyde_logger:
        hyde_logger.info(f"Question: {text_wrap(query)}")
        hyde_logger.info(f"Hypothetical Document: {text_wrap(hyde_output)}")
        hyde_logger.info(f"Hyde Query: {text_wrap(hyde_query)}")

    return hyde_query

# Output processing functions
def clean_output(output):
    """Returns LLM's answer from its output
    Assumes output is in JSON format of {'Answer':answer}.
    If output is not in expected format, returns string signifying error
    """
    end_index = output.find('}')
    if end_index == -1:
        end_index = len(output)
    json_text = output[:end_index+1].strip()
    try:
        d = json.loads(json_text)
        return str(d['Answer'])
    except:
        return f'JSON parse error({json_text})'

# Data access functions
def get_question(i):
    """Get question of i-th entry in data."""
    example = data[i]
    example_q = example['question']
    input = preprocess_text(example_q)
    return input

def get_answer(i):
    """Get answer of i-th entry in data."""
    example = data[i]
    return example['answer']

# Document processing functions
def remove_duplicate_docs(docs):
    """Returns list of Documents without the duplications. Order preserved."""
    results = []
    for doc in docs:
        if any(map(lambda d:duplicate(d, doc), results)): continue
        results.append(doc)
    return results

# Retriever utility functions
def set_retriever_k(retriever, k):
    if isinstance(retriever, BM25Retriever):
        retriever.k = k
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

# Evaluation metric functions
def included_ratio(answer, docs):
    """Returns if answer is in one of the documents. 1 if true, 0 if false."""
    return int(any(answer in doc.page_content for doc in docs))

# Main evaluation functions
def eval_full_chain(retriever, k, num_of_tests=len(data), shuffle=False, rerank=False, verbose=False, 
                   input_path=input_file_path, output_path=output_file_path, hyde=False, 
                   eval_logger=None, hyde_logger=None):
    separator = "-" * 50

    if eval_logger:
        eval_logger.info(f"Starting evaluation: k={k}, num_of_tests={num_of_tests}, hyde={hyde}")
    
    print('Creating Retrieval Chain...')
    if rerank:
        retrieval_chain = get_retrieval_chain_w_rerank(retriever, k)
    else:
        retrieval_chain = get_retrieval_chain(retriever, k, shuffle)
    
    print('Create Input...')
    inputs = list(map(get_question, range(num_of_tests)))
    print('Running Inference...')
    outputs = []
    
    range_iter = tqdm(range(num_of_tests), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}') if not verbose else range(num_of_tests)

    for i in range_iter:
        input = inputs[i]
        if eval_logger:
            eval_logger.info(separator)
            eval_logger.info(f"Question {i}: {text_wrap(input)}")

        chain_output = retrieval_chain(input)
        outputs.append(chain_output)
        if verbose:
            print(f'Question: {input}')
            print(f'Answer: {get_answer(i)}')
            print(f'Generated: {clean_output(chain_output)}')
            print()
        if eval_logger:
            eval_logger.info(f"Generated Answer: {text_wrap(clean_output(chain_output))}")
            eval_logger.info(f"Correct Answer: {text_wrap(get_answer(i))}")

    print('Cleaning Output...')
    outputs = list(map(clean_output, outputs))
    print('Evaluating...')
    result = []
    for i in range(num_of_tests):
        ans = get_answer(i)
        out = outputs[i]
        result.append({'question':inputs[i], 'answers':[ans], 'generated_answer':[out]})

    result_str = list(map(lambda x:json.dumps(x, ensure_ascii=False), result))
    with open(input_path, 'w+', encoding='utf-8') as f:
        f.write('\n'.join(result_str))

    evaluate_by_dicts(input_path, output_path)
    with open(output_path, 'r', encoding='utf-8') as f:
        evaluations = json.load(f)
    llm_eval = RAG_eval_w_LLM(input_path, llm_output_file_path)
    evaluations['llm'] = llm_eval

    if eval_logger:
        eval_logger.info(separator)
        eval_logger.info(f"Evaluation complete. Final result: {evaluations}")
    
    return evaluations

def eval_retriever(retriever, k, num_of_tests=len(data), hyde=False, eval_logger=None, hyde_logger=None):
    """Returns the evaluation of retriever alone, by the context it retrieves.
    Basically the average of included_ratio function for the test cases.
    Always simply picks the entries at front.
    """
    ratios = []
    separator = "-" * 50

    print('Evaluating...')
    if eval_logger:
        eval_logger.info(f"Starting evaluation: k={k}, num_of_tests={num_of_tests}, hyde={hyde}")
        eval_logger.info(separator) 

    if not hyde:
        for i in tqdm(range(num_of_tests), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            context = get_k_from_retriever(retriever, k, get_question(i))
            ans = get_answer(i)
            ratio = included_ratio(ans, context)
            ratios.append(ratio)

            if eval_logger:
                eval_logger.info(f"Question {i}: Ratio = {ratio}")
                eval_logger.info(separator)

    else:
        for i in tqdm(range(num_of_tests), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            question = get_question(i)
            formatted_prompt = format_hyde_prompt(question, chunk_size=500)
            hyde_output = hyde_generate(formatted_prompt, chunk_size=500)
            hyde_query = (question + '\n')*4 + hyde_output
            if hyde_logger:
                hyde_logger.info(f"Question {i}: {text_wrap(question)}")
                hyde_logger.info(f"Hypothetical Document {i}: {text_wrap(hyde_output)}")
                hyde_logger.info(f"Hyde Query {i}: {text_wrap(hyde_query)}")
                hyde_logger.info(separator)

            context = get_k_from_retriever(retriever, k, hyde_query)
            ans = get_answer(i)
            ratio = included_ratio(ans, context)
            ratios.append(ratio)

            if eval_logger:
                eval_logger.info(f"Question {i}: Ratio = {ratio}")
                eval_logger.info(separator)

    final_ratio = sum(ratios)/num_of_tests
    if eval_logger:
        eval_logger.info(f"Evaluation complete. Final ratio: {final_ratio}")
    return final_ratio
