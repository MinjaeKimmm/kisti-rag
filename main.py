import os
import shutil

import chromadb
import json

from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

from datetime import datetime

from pipeline.chunking.simple import get_simple_retriever
from pipeline.chunking.sentence_parent import get_sentence_parent_retriever, delete_sentence_parent_retriever_data
from pipeline.infer import eval_retriever, invoke_retriever, get_answer, get_question, eval_retriever_on, eval_full_chain, get_k_from_retriever, hyde_query_generate
from pipeline.util.embed import delete_embeddings, remove_small_chunks
from pipeline.common import embedding_function, embedding_dir, setup_logger
from pipeline.eval.eval import input_file_path, output_file_path, evaluate_by_dicts, recalculate_metrics
from pipeline.util.dense_runnable import DenseRetrieverWithHyde
from pipeline.common import input_path, output_path

DENSE='dense'
SPARSE='sparse'
ENSEMBLE='ensemble'

def top_k_experiment(retriever_type, hyde):
    base_subdirectory = f'eval_full_chain/{retriever_type}'
    eval_logger = setup_logger('eval_full_chain', subdirectory=base_subdirectory)
    hyde_logger = setup_logger('hyde', subdirectory=base_subdirectory) if hyde==True else None

    ks = [8]

    sparse_retriever = get_simple_retriever('bm25', 500, 50)
    dense_retriever = DenseRetrieverWithHyde(get_sentence_parent_retriever(500, 125), hyde=hyde, hyde_logger=hyde_logger)

    retriever_map = {
        'dense': dense_retriever,
        'sparse': sparse_retriever,
        'ensemble': EnsembleRetriever(
            retrievers=[dense_retriever, sparse_retriever],
            weights=[0.5, 0.5]
        )
    }

    if retriever_type not in retriever_map:
        raise ValueError(f"Unknown retriever type: {retriever_type}.")
    
    for k in ks:
        result=eval_full_chain(retriever_map[retriever_type], k, input_path=input_path, output_path=output_path, hyde=hyde, eval_logger=eval_logger, hyde_logger=hyde_logger)
        print(k, result)

if __name__ == '__main__':
    top_k_experiment(DENSE, False)
