import cmd
import json
import argparse
from datetime import datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain.retrievers import EnsembleRetriever

from pipeline.infer.infer import get_retrieval_chain, clean_output, get_retrieval_chain_w_rerank
from pipeline.chunking.simple import get_simple_retriever
from pipeline.chunking.sentence_parent import get_sentence_parent_retriever
from pipeline.util.dense_runnable import DenseRetrieverWithHyde

DENSE = 'dense'
SPARSE = 'sparse'
ENSEMBLE = 'ensemble'

class RAGCLI(cmd.Cmd):
    intro = 'Welcome to the RAG testing environment. Type help or ? to list commands.\n'
    prompt = '(kisti-rag) '
    
    def __init__(self, retriever_type='dense', k=3, hyde=False, use_rerank=False):
        super().__init__()
        self.retriever_type = retriever_type
        self.k = k
        self.hyde = hyde
        self.use_rerank = use_rerank
        self.results_file = f'results/interactive_qa.json'
        
        self.data = {
            'config': {
                'retriever': self.retriever_type,
                'k': self.k,
                'hyde': self.hyde,
                'rerank': self.use_rerank
            },
            'results': []
        }
        self.setup_retrievers()
        
    def setup_retrievers(self):
        """Initialize retrievers based on configuration"""
        print(f"Initializing {self.retriever_type} retriever (k={self.k}, hyde={self.hyde}, rerank={self.use_rerank})...")
        
        self.sparse_retriever = get_simple_retriever('bm25', 500, 50)
        self.dense_retriever = DenseRetrieverWithHyde(
            get_sentence_parent_retriever(500, 125),
            hyde=self.hyde
        )
        
        self.retriever_map = {
            'sparse': self.sparse_retriever,
            'dense': self.dense_retriever,
            'ensemble': EnsembleRetriever(
                retrievers=[self.dense_retriever, self.sparse_retriever],
                weights=[0.5, 0.5]
            )
        }
        
        if self.retriever_type not in self.retriever_map:
            raise ValueError(f"Unknown retriever type: {self.retriever_type}")
            
        retriever = self.retriever_map[self.retriever_type]
        if self.use_rerank:
            self.chain = get_retrieval_chain_w_rerank(retriever, self.k)
        else:
            self.chain = get_retrieval_chain(retriever, self.k)
            
        print("Initialization complete! Type 'query <your question>' to start.")
    
    def do_query(self, query):
        """Query the RAG system: query <your question>"""
        if not query:
            print("Please provide a query. Usage: query <your question>")
            return
            
        raw_output = self.chain(query)
        cleaned_output = clean_output(raw_output)
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'question': query,
            'generated_answer': cleaned_output
        }
        self.data['results'].append(result)
        
        Path('results').mkdir(exist_ok=True)
        with open(self.results_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print(cleaned_output)
    
    def do_config(self, arg):
        """Show current configuration"""
        print(f"\nCurrent configuration:")
        print(f"Retriever type: {self.retriever_type}")
        print(f"k: {self.k}")
        print(f"HyDE: {self.hyde}")
        print(f"Rerank: {self.use_rerank}")
        print(f"Results file: {self.results_file}")
    
    def do_exit(self, arg):
        """Exit the RAG testing environment"""
        print("Goodbye!")
        return True
        
    def default(self, line):
        if line.strip() == 'EOF':
            return self.do_exit(line)
        print(f"Unknown command: {line}\nType 'help' or '?' to see available commands.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive RAG Testing Environment')
    parser.add_argument('--retriever', choices=[DENSE, SPARSE, ENSEMBLE], default=DENSE,
                      help='Retriever type to use')
    parser.add_argument('--k', type=int, default=4,
                      help='Number of documents to retrieve')
    parser.add_argument('--hyde', action='store_true',
                      help='Whether to use HyDE')
    parser.add_argument('--rerank', action='store_true',
                      help='Whether to use reranking')
    
    args = parser.parse_args()
    
    try:
        cli = RAGCLI(
            retriever_type=args.retriever,
            k=args.k,
            hyde=args.hyde,
            use_rerank=args.rerank
        )
        cli.cmdloop()
    except KeyboardInterrupt:
        print("\nGoodbye!")
