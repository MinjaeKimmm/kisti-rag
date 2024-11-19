from ..common import text_wrap
from ..llm import format_hyde_prompt, hyde_generate

def hyde_query_generate(query, hyde, hyde_logger=None):
    if not hyde:
        return query  # Return the original query if hyde is not enabled

    # Create the hypothetical document generation
    formatted_prompt = format_hyde_prompt(query, chunk_size=500)
    hyde_output = hyde_generate(formatted_prompt, chunk_size=500)
    hyde_query = (query + '\n') * 4 + hyde_output[:125]
    
    # Log the hypothetical document if logging is enabled
    if hyde_logger:
        hyde_logger.info(f"Question: {text_wrap(query)}")
        hyde_logger.info(f"Hypothetical Document: {text_wrap(hyde_output)}")
        hyde_logger.info(f"Hyde Query: {text_wrap(hyde_query)}")

    return hyde_query