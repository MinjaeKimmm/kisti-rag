import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .common import get_free_gpu, model_name

free_gpu = get_free_gpu()
device = torch.device(f'cuda:{free_gpu}')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

def format_prompt(query, retrieved_documents):
    PROMPT = f"""
Based on the given reference documents, answer the following question.
When answering, do not repeat the question, and only provide the correct answer in korean.
Provide the answer only in JSON format as {{"Answer":"Your answer"}}.
Answer should be Korean.
Reference Documents:
---------------------
{retrieved_documents}
——————————
Question: {query}
Answer: Korean
"""
    return PROMPT

def generate(formatted_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": formatted_prompt}  
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=6000,
        temperature = 1e-10,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

def format_hyde_prompt(query, chunk_size):
    PROMPT = f"""Generate a section of a Korean academic paper that addresses the following question: '{query}'
    This content should:

    1. Be exactly {chunk_size} characters in length.
    2. Use formal, academic Korean language.
    3. Include field-specific technical terms and jargon in Korean(English or Chinese terms may be used if necessary).
    4. Provide explanations of key concepts.
    5. Maintain a neutral and objective tone.
    6. Offer information from a global context (not limited to a specific country).
    7. Start directly with the content without using prefixes, repeating the question, or explicitly answering it.
    9. End your response when you finish creating the Korean text. Do not produce any further explanations, translations, paraphrasing in English. Do not explain the Korean text or explain it adheres to the guidline. Do not give me any translations. End your response with the Korean text.

    The Korean text should resemble a genuine excerpt from an academic paper. Following these guidelines, provide a detailed and in-depth Korean content in a single paragraph. """

    return PROMPT

def hyde_generate(formatted_prompt, chunk_size):
    input_ids = tokenizer(formatted_prompt, return_tensors="pt").input_ids.to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens = chunk_size + 50,
            temperature = 1e-10,
        )
    generated_ids = output_ids[0, input_ids.shape[1]:]
    hypothetical_doc = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return hypothetical_doc
