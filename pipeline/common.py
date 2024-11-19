# Standard library imports
import os
import re
import logging
import textwrap
from datetime import datetime

# Third-party imports
import numpy as np
import torch
from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# Constants and configurations
embedding_model = 'intfloat/multilingual-e5-large-instruct'
embedding_dir = '/data3/temp/embed'
log_dir = '/data3/temp/logs'
model_name = 'Qwen/Qwen2-7B-Instruct'
input_path = 'results/generated_answer.json'
output_path = 'results/evaluation.json'

# Initialize Kiwi
kiwi = Kiwi()

# GPU Setup
def get_free_gpu():
    """In Linux kernel, type export CUDA_VISIBLE_DEVICES='[list of gpu numbers]'"""
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return None

    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    
    if cuda_visible_devices:
        gpu_ids = [int(x) for x in cuda_visible_devices.split(',') if x.strip()]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"Available GPUs: {gpu_ids}")
    
    if not gpu_ids:
        print("No GPUs available.")
        return None

    free_memory = []
    for i, gpu_id in enumerate(gpu_ids):
        try:
            torch.cuda.set_device(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory
            free_mem, _ = torch.cuda.mem_get_info(i)
            free_memory.append(free_mem)
            print(f"GPU {gpu_id}: {free_mem/1024**3:.2f} GB free / {total_memory/1024**3:.2f} GB total")
        except Exception as e:
            print(f"GPU {gpu_id}: Unable to query device properties - {str(e)}")
            free_memory.append(0)

    if not free_memory or max(free_memory) == 0:
        print("Unable to query GPU memory or all GPUs are full.")
        return None

    selected_index = free_memory.index(max(free_memory))
    selected_gpu = gpu_ids[selected_index]
    print(f"Selected GPU: {selected_gpu}")
    
    return selected_index

# Set up device and embedding function
device = torch.device(f'cuda:{get_free_gpu()}')
embedding_function = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'device':device})

# Text processing functions
def text_wrap(text, width=100, indent=''):
    """Wrap text to a specified width and optionally add an indent."""
    wrapped_lines = textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)
    return '\n'.join(wrapped_lines)

def preprocess_text(text):
    """Standard method for preprocessing input query"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def remove_metadata(text):
    """Remove metadata from chunk text. Returns without modification if it does not have metadata attached."""
    cut_string = 'content: '
    if cut_string in text:
        index = text.find(cut_string)+len(cut_string)
        text = text[index:]
    return text

def kiwi_tokenizer(text):
    return [token.form for token in kiwi.tokenize(text)]

# Utility functions
def duplicate(d1, d2):
    """Check if two documents have the same page content."""
    return d1.page_content == d2.page_content

def setup_logger(name, log_dir=log_dir, subdirectory=''):
    """Set up Logger"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    datetime_folder = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    full_log_dir = os.path.join(log_dir, subdirectory, datetime_folder)
    os.makedirs(full_log_dir, exist_ok=True)

    log_filename = f"{name}.log"
    
    file_handler = logging.FileHandler(os.path.join(full_log_dir, log_filename))
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger