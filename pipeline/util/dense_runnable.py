from langchain_core.runnables.base import Runnable
from ..infer.hyde import hyde_query_generate

class DenseRetrieverWithHyde(Runnable):
    def __init__(self, dense_retriever, hyde=False, hyde_logger=None):
        self.dense_retriever = dense_retriever
        self.hyde = hyde
        self.hyde_logger = hyde_logger

    def invoke(self, query, run_manager=None):
        modified_query = hyde_query_generate(query, self.hyde, self.hyde_logger)
        return self.dense_retriever.invoke(modified_query, run_manager=run_manager)
    
    @property
    def search_kwargs(self):
        return self.dense_retriever.search_kwargs

    @search_kwargs.setter
    def search_kwargs(self, value):
        self.dense_retriever.search_kwargs = value
