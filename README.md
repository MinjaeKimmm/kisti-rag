# KISTI-RAG

This repository contains the code for a joint research conducted by KAIST DM Lab and KAIST DISL Lab in collaboration with KISTI (Korea Institute of Science and Technology Information) to develop an advanced RAG system for Korea's largest scientific literature database, focusing on optimizing large-scale information retrieval and question answering specific to Korean literature.

## Project Structure

```
├── data/            # Contains Korean academic paper dataset and QA pairs
├── embed/           # Storage for document embeddings and retrieval indexes
├── logs/            # Experiment logs including retrieval and evaluation results
├── pipeline/
│   ├── util/        # Utility functions for data processing and embedding
│   ├── chunking/    # Document chunking strategies and retriever implementations
│   ├── infer/       # Core RAG pipeline components: query processing, retrieval, and generation
│   ├── eval/        # Evaluation framework with multiple metrics for QA assessment
│   ├── common.py    # Shared constants, configurations, and utility functions
│   └── llm.py       # LLM configuration and prompt templates
├── results/         # Evaluation results and performance metrics
└── main.py         # Entry point for running experiments
```

## Setup Requirements

### Dataset Setup

1. Download both datasets from [KISTI AIDA](https://aida.kisti.re.kr/data/?collection=&q=%EB%85%BC%EB%AC%B8):

   - 국내 논문 전문 텍스트 데이터셋
   - 국내 논문 QA 데이터셋
2. Place both dataset folders in `data/KISTI/`

### Installation

```bash
pip install -r requirements.txt
```

## Running Experiments

There are two ways to run experiments:

### 1. Batch Experiments

Execute the main script:

```bash
python main.py
```

The main script (`main.py`) provides three primary functions:

1. **Create Sample Data**

   ```python
   create_sample_data(sample_paper_num, sample_qa_num)
   ```

   - Samples data from both the corpus and the QA dataset
   - Parameters:
       - `sample_paper_num`: Number of papers to sample
       - `sample_qa_num`: Number of QA pairs to sample
2. **Retrieval Chain Evaluation**

   ```python
   retrieval_chain(retriever_type, hyde)
   ```

   - Evaluates the retrieval performance with LLM generation
   - Parameters:
       - `retriever_type`: One of `DENSE`, `SPARSE`, or `ENSEMBLE`
       - `hyde`: Boolean, applies to dense retriever in both `DENSE` and `ENSEMBLE` modes
3. **Full Chain Evaluation**

   ```python
   full_chain(retriever_type, hyde, rerank=False)
   ```

   - Evaluates full RAG pipeline with LLM generation and evaluation
   - Parameters:
       - `retriever_type`: One of `DENSE`, `SPARSE`, or `ENSEMBLE`
       - `hyde`: Boolean, applies to dense retriever
       - `rerank`: Boolean, whether to use cross-encoder reranking

Example usage in main.py:

```python
if __name__ == '__main__':
    #create_sample_data(3,30)
    #retrieval_chain(ENSEMBLE, True)
    full_chain(DENSE, False, rerank=True)
```

### 2. Interactive Mode

For testing individual queries interactively:

```bash
python interactive.py [options]
```

Options:

- `--retriever`: Choose retriever type (`DENSE`, `SPARSE`, `ENSEMBLE`)
- `--k`: Number of documents to retrieve (default: 4)
- `--hyde`: Enable Hypothetical Document Embeddings
- `--rerank`: Enable cross-encoder reranking

In the interactive shell:

- Type `query your question here` to test queries
- Use `config` to see current settings
- Type `exit` or press Ctrl+D to quit

Results are saved in `results/interactive_qa.json` with configuration and query history.

## Code Documentation

### Core Components

1. **Common Utilities** (`pipeline/common.py`)
   ```python
   from pipeline.common import device, embedding_function, setup_logger, preprocess_text
   ```
   - Core configurations and utilities
   - Components:
       - GPU management and device setup
       - Embedding model configuration (`multilingual-e5-large-instruct`)
       - Text preprocessing with Kiwi tokenizer
   - Utility Functions:
       - `setup_logger()`: Configures logging with timestamps
       - `preprocess_text()`: Standardizes input text
       - `text_wrap()`: Formats text output
       - `remove_metadata()`: Cleans document chunks

2. **LLM Integration** (`pipeline/llm.py`)
   ```python
   from pipeline.llm import format_prompt, generate, format_hyde_prompt, hyde_generate
   ```
   - Core LLM integration (Qwen2-7B-Instruct)
   - Functions:
       - `format_prompt()`: Creates QA prompt with context
       - `generate()`: Generates answer using LLM
       - `format_hyde_prompt()`: HyDE prompt formatting
       - `hyde_generate()`: HyDE document generation

3. **Document Processing**

   a. Simple Chunking (`chunking/simple.py`)
   ```python
   from pipeline.chunking.simple import get_simple_retriever, get_simple_splitter
   ```
   - Fixed-size document chunking
   - Functions:
       - `get_simple_splitter()`: Creates text splitter
       - `get_simple_retriever()`: Creates retriever (dense/BM25)
   - Example:
     ```python
     retriever = get_simple_retriever('dense', chunk_size=500, chunk_overlap=50)
     ```

   b. Sentence-Parent Chunking (`chunking/sentence_parent.py`)
   ```python
   from pipeline.chunking.sentence_parent import get_sentence_parent_retriever
   ```
   - Hierarchical document chunking
   - Better context preservation
   - Example:
     ```python
     retriever = get_sentence_parent_retriever(parent_size=1000, child_size=200)
     ```

4. **Embedding Management** (`util/embed.py`)
   ```python
   from pipeline.util.embed import embed, delete_embeddings, remove_small_chunks
   ```
   - Document embedding management
   - Functions:
       - `embed()`: Creates embedding collection
       - `delete_embeddings()`: Removes collection
       - `remove_small_chunks()`: Filters chunks by size

5. **Data Processing** (`util/kisti_data.py`)
   ```python
   from pipeline.util.kisti_data import get_sample_paper, get_sample_qa
   ```
   - KISTI dataset handling
   - Paper and QA pair sampling

### Retrieval and Inference Pipeline

1. **Document Retrieval** (`infer/retriever.py`)
   ```python
   from pipeline.infer.retriever import get_k_from_retriever
   ```
   - Document retrieval interface
   - Handles both dense and sparse retrieval
   - Example:
     ```python
     # Initialize retriever
     retriever = get_simple_retriever('dense', 500, 50)
     
     # Retrieve documents
     docs = get_k_from_retriever(retriever, k=4, query="연구의 주요 기여는?")
     ```

2. **Retrieval Chain** (`infer/infer.py`, `infer/reranking.py`)
   ```python
   from pipeline.infer.infer import get_retrieval_chain
   from pipeline.infer.reranking import get_retrieval_chain_w_rerank
   ```
   - Complete RAG pipeline
   - Supports optional reranking
   - Example:
     ```python
     # Basic chain
     chain = get_retrieval_chain(retriever, k=4)
     
     # Chain with reranking
     chain_rerank = get_retrieval_chain_w_rerank(retriever, k=4)
     ```

### Evaluation Framework

1. **Core Evaluation** (`eval/eval.py`)
   ```python
   from pipeline.eval.eval import evaluate_by_dicts, recalculate_metrics
   ```
   - Comprehensive evaluation suite
   - Metrics:
       - Answer EM/F1 scores
       - BLEU and ROUGE scores
       - BERTScore for semantic similarity

2. **LLaMA-3 Evaluation** (`eval/llama3_eval.py`)
   ```python
   from pipeline.eval.llama3_eval import RAG_eval_w_LLM
   ```
   - LLaMA-3 based answer evaluation
   - Components:
       - 8B Instruct model for scoring
       - System prompt for fair evaluation
       - Score extraction from LLM output
   - Functions:
       - `RAG_eval_w_LLM(eval_path, output_path)`: Run LLM evaluation
       - `generate(formatted_prompt)`: Generate LLM response
       - `format_query_prompt(query, gt, pd)`: Format evaluation prompt
   - Example:
     ```python
     RAG_eval_w_LLM(
         'results/eval_input.json',
         'results/llm_evaluation.json'
     )
     ```

3. **Evaluation Utilities** (`eval/lib.py`)
   ```python
   from pipeline.eval.lib import read_json, write_json, read_jsonl, write_jsonl
   ```
   - Support functions for evaluation
   - Components:
       - JSON/JSONL file handling
       - Dataset inference
       - Server configuration
   - Functions:
       - File I/O: `read_json`, `write_json`, `read_jsonl`, `write_jsonl`
       - Config: `get_llm_server_address`, `get_retriever_address`
       - Dataset: `infer_dataset_from_file_path`
       - Text: `find_matching_paragraph_text`

4. **Specialized Metrics** (`eval/metrics/`)
   ```python
   from pipeline.eval.metrics import (
       DropAnswerEmAndF1,
       SupportEmF1Metric,
       AnswerSupportRecallMetric,
       SquadAnswerEmF1Metric
   )
   ```
   - Task-specific evaluation metrics
   - Support for various QA formats

5. **Example Usage**
   ```python
   # Initialize components
   retriever = get_simple_retriever('dense', 500, 50)
   chain = get_retrieval_chain(retriever, k=4)
   
   # Run evaluation
   results = eval_full_chain(
       retriever,
       k=4,
       rerank=True,
       hyde=True,
       verbose=True
   )
   
   # LLM evaluation
   llm_scores = RAG_eval_w_LLM(
       'results/eval_input.json',
       'results/llm_evaluation.json'
   )
   ```

### Inference Components

1. **Core Inference** (`infer/infer.py`)
   ```python
   from pipeline.infer.infer import get_retrieval_chain, eval_full_chain, clean_output
   ```
   - Core RAG inference functionality
   - Functions:
       - `get_retrieval_chain(retriever, k, shuffle=False)`: Creates retrieval chain function
       - `eval_full_chain(retriever, k, ...)`: Full evaluation pipeline with metrics
       - `clean_output(output)`: Cleans LLM output from JSON format
   - Parameters:
       - `retriever`: Document retriever instance
       - `k`: Number of documents to retrieve
       - `shuffle`: Optional document shuffling
       - `rerank`: Use cross-encoder reranking
       - `hyde`: Use HyDE for query expansion
       - `verbose`: Print detailed progress
   - Example:
     ```python
     chain = get_retrieval_chain(retriever, k=4)
     results = eval_full_chain(
         retriever,
         k=4,
         rerank=True,
         hyde=True,
         verbose=True
     )
     ```

2. **HyDE Integration** (`infer/hyde.py`)
   ```python
   from pipeline.infer.hyde import format_hyde_prompt
   ```
   - Hypothetical Document Embedding support
   - Expands queries using generated hypothetical documents

3. **Reranking** (`infer/reranking.py`)
   ```python
   from pipeline.infer.reranking import get_retrieval_chain_w_rerank
   ```
   - Cross-encoder reranking of retrieved documents
   - Improves retrieval precision

### Evaluation Framework

1. **Core Evaluation** (`eval/eval.py`)
   ```python
   from pipeline.eval.eval import evaluate_by_dicts, recalculate_metrics
   ```
   - Main evaluation pipeline
   - Metrics:
       - Answer EM/F1 scores
       - BLEU and ROUGE scores
       - BERTScore
   - Functions:
       - `evaluate_by_dicts()`: Main evaluation pipeline
       - `recalculate_metrics()`: Updates metrics

2. **LLaMA-3 Evaluation** (`eval/llama3_eval.py`)
   ```python
   from pipeline.eval.llama3_eval import RAG_eval_w_LLM
   ```
   - LLaMA-3 based answer evaluation
   - Components:
       - 8B Instruct model for scoring
       - System prompt for fair evaluation
       - Score extraction from LLM output
   - Functions:
       - `RAG_eval_w_LLM(eval_path, output_path)`: Run LLM evaluation
       - `generate(formatted_prompt)`: Generate LLM response
       - `format_query_prompt(query, gt, pd)`: Format evaluation prompt
   - Example:
     ```python
     RAG_eval_w_LLM(
         'results/eval_input.json',
         'results/llm_evaluation.json'
     )
     ```

3. **Evaluation Utilities** (`eval/lib.py`)
   ```python
   from pipeline.eval.lib import read_json, write_json, read_jsonl, write_jsonl
   ```
   - Support functions for evaluation
   - Components:
       - JSON/JSONL file handling
       - Dataset inference
       - Server configuration
   - Functions:
       - File I/O: `read_json`, `write_json`, `read_jsonl`, `write_jsonl`
       - Config: `get_llm_server_address`, `get_retriever_address`
       - Dataset: `infer_dataset_from_file_path`
       - Text: `find_matching_paragraph_text`

4. **Specialized Metrics** (`eval/metrics/`)
   ```python
   from pipeline.eval.metrics import (
       DropAnswerEmAndF1,
       SupportEmF1Metric,
       AnswerSupportRecallMetric,
       SquadAnswerEmF1Metric
   )
   ```
   - Task-specific evaluation metrics
   - Support for various QA formats

5. **Example Usage**
   ```python
   # Initialize components
   retriever = get_simple_retriever('dense', 500, 50)
   chain = get_retrieval_chain(retriever, k=4)
   
   # Run evaluation
   results = eval_full_chain(
       retriever,
       k=4,
       rerank=True,
       hyde=True,
       verbose=True
   )
   
   # LLM evaluation
   llm_scores = RAG_eval_w_LLM(
       'results/eval_input.json',
       'results/llm_evaluation.json'
   )
   ```