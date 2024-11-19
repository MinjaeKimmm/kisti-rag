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

Execute the main script:
```bash
python -m main
```

The main script (`main.py`) serves as the entry point for running experiments and provides three primary functions:

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
   full_chain(retriever_type, hyde)
   ```
   - Evaluates full RAG pipeline with LLM generation and evaluation
   - Parameters same as `retrieval_chain`

Example usage in main.py:
```python
if __name__ == '__main__':
    #create_sample_data(3,30)
    #retrieval_chain(ENSEMBLE, True)
    full_chain(ENSEMBLE, False)
```
