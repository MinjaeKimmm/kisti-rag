from langchain_core.runnables.base import Runnable

class DenseRetrieverWithHyde(Runnable):
    def __init__(self, dense_retriever, hyde=False, hyde_logger=None):
        self.dense_retriever = dense_retriever
        self.hyde = hyde
        self.hyde_logger = hyde_logger

    def invoke(self, query, run_manager=None):
        # Apply hyde logic if enabled
        modified_query = hyde_query_generate(query, self.hyde, self.hyde_logger)
        # Use the dense retriever with the modified or original query
        return self.dense_retriever.invoke(modified_query, run_manager=run_manager)
    
    @property
    def search_kwargs(self):
        # Pass through the search_kwargs to the underlying dense retriever
        return self.dense_retriever.search_kwargs

    @search_kwargs.setter
    def search_kwargs(self, value):
        # Allow setting search_kwargs on the underlying dense retriever
        self.dense_retriever.search_kwargs = value
