import os
import json
import shutil
from pathlib import Path

from tqdm import tqdm

from langchain.docstore.document import Document
from ..common import preprocess_text

data_folder = 'data/KISTI'
sample_folder = 'data/KISTI_sample'
paper_folder_name = '(분류) 국내 논문 전문 텍스트 데이터셋'
qa_folder_name = '(분류) 국내 논문 QA 데이터셋'

def ensure_directories_exist():
    """Create necessary directories if they don't exist"""
    directories = [
        os.path.join(data_folder, paper_folder_name),
        os.path.join(data_folder, qa_folder_name),
        os.path.join(sample_folder, paper_folder_name),
        os.path.join(sample_folder, qa_folder_name)
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

"""Returns path and metadata for all files under the folder in list of dicts.

Assumes the folder structure is like KISTI data (journal>year>papers).
Dicts in {'path':..., 'metadata':{'journal':..., 'year':...}} format.
"""
def get_file_paths(folder_path):
    print('Getting File Paths...')
    files = []
    
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(folder_path) or not os.listdir(folder_path):
        print(f"Warning: Directory {folder_path} is empty or doesn't exist")
        return files
        
    journals = sorted(os.listdir(folder_path))
    for journal in tqdm(journals):
        journal_path = os.path.join(folder_path, journal)
        if not os.path.isdir(journal_path):
            continue
        years = sorted(os.listdir(journal_path))
        for year in years:
            year_path = os.path.join(journal_path, year)
            if not os.path.isdir(year_path):
                continue
            papers = sorted(os.listdir(year_path))
            for paper in papers:
                paper_path = os.path.join(year_path, paper)
                files.append({'path':paper_path, 'metadata':{'journal':journal, 'year':year}})
    return files

"""Returns all papers in folder as list of Langchain Documents"""
def get_paper(folder):
    folder_path = os.path.join(folder, paper_folder_name)
    files = get_file_paths(folder_path)
    print('Getting Papers Data...')
    data = []
    for file in tqdm(files):
        data.extend(get_paper_data(file['path'], file['metadata']))
    return data

def get_full_paper():
    return get_paper(data_folder)
def get_sample_paper():
    return get_paper(sample_folder)

"""Returns all QA data in folder as list of dicts.

Dicts in {'level':..., 'question':..., 'answer':...} format.
"""
def get_qa(folder):
    folder_path = os.path.join(folder, qa_folder_name)
    files = get_file_paths(folder_path)
    print('Getting QA Data...')
    data = []
    for file in tqdm(files):
        data.extend(get_qa_data(file['path']))
    return data

def get_full_qa():
    return get_qa(data_folder)
def get_sample_qa():
    return get_qa(sample_folder)

"""Chooses and returns string from dictionary of multiple languages. Prefers Korean."""
def choose_language(d):
    if 'ko' in d:
        return d['ko']
    if 'en' in d:
        return d['en']
    return ''

"""Returns contents of paper in list of Langchain Documents.

metadata: metadata that should be applied to all the documents created.
"""
def get_paper_data(file_path, metadata):
    data = []
    with open(file_path, encoding='utf8') as f:
        content = json.load(f)
    if 'doc_id' in content:
        metadata = metadata|{'doc_id':content['doc_id']}
    if 'title' in content:
        title_text = choose_language(content['title'])
        metadata = metadata|{'title':title_text}
    if 'authors' in content:
        metadata = metadata|{'authors':content['authors']}
    if 'keywords' in content:
        keywords_text = choose_language(content['keywords'])
        metadata = metadata|{'keywords':keywords_text}
    if 'abstract' in content:
        abstract_text = choose_language(content['abstract'])
        if abstract_text:
            abstract_doc = Document(abstract_text, metadata=metadata|{'section':'abstract'})
            data.append(abstract_doc)
    body = content['body_text']
    for section in body:
        if 'text' not in section: continue
        add_metadata = {}
        if 'section' in section:
            add_metadata['section'] = section['section']
        doc = Document('\n'.join(section['text']), metadata=metadata|add_metadata)
        data.append(doc)
    return data

"""Returns contents of paper in single string."""
def get_paper_data_in_string(file_path):
    docs = get_paper_data(file_path, {})
    contents = [doc.page_content for doc in docs]
    full_str = '\n'.join(contents)
    return full_str

"""Returns contents of QA file as list of dicts.

Dicts in {'level':..., 'question':..., 'answer':...} format.
"""
def get_qa_data(file_path):
    data = []
    with open(file_path, encoding='utf8') as f:
        content = json.load(f)
    if 'qas' not in content: return []
    for qa in content['qas']:
        data.append({'level':qa['level'], 'question':qa['question'], 'answer':qa['answer']['answer_text']})
    return data

def document_to_dict(doc):
    return {'page_content':doc.page_content, 'metadata':doc.metadata}

"""Returns whether file should be included in sampling.

Attempts to excludes papers in English.
"""
def should_be_sampled(file_path):
    text = get_paper_data_in_string(file_path)
    if len(text) == 0: return False
    alphabets_count = sum(c.encode().isalpha() for c in text)
    alphabets_ratio = alphabets_count/len(text)
    if alphabets_ratio > 0.5: return False
    return True

"""Copies portion of KISTI data to sample folder.

Attempts to copy given number of papers for each journal, unless there are not enough papers.
Uses alphabetical ordering to select papers, may exclude certain files(see should_be_sampled function).
"""
def sample_kisti(sample_paper_num=3, sample_qa_num=3):
    print('Removing existing samples...')
    shutil.rmtree(sample_folder, ignore_errors=True)
    print('Sampling data...')
    paper_folder = os.path.join(data_folder, paper_folder_name)
    journals = sorted(os.listdir(paper_folder))

    def sample_journal(journal):
        journal_path = os.path.join(paper_folder, journal)
        years = sorted(os.listdir(journal_path))
        sampled_count = 0
        for year in years:
            year_path = os.path.join(paper_folder, journal, year)
            papers = sorted(os.listdir(year_path))
            for paper in papers:
                if sampled_count < sample_paper_num:
                    old_paper_path = os.path.join(data_folder, paper_folder_name, journal, year, paper)
                    new_paper_path = os.path.join(sample_folder, paper_folder_name, journal, year, paper)
                    if not should_be_sampled(old_paper_path): continue
                    Path(new_paper_path).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(old_paper_path, new_paper_path)
                
                if sampled_count < sample_qa_num:
                    old_qa_path = os.path.join(data_folder, qa_folder_name, journal, year, paper)
                    new_qa_path = os.path.join(sample_folder, qa_folder_name, journal, year, paper)
                    if os.path.exists(old_qa_path):
                        Path(new_qa_path).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(old_qa_path, new_qa_path)
                
                if sampled_count >= sample_paper_num and sampled_count >= sample_qa_num:
                    return
                sampled_count += 1

    for journal in tqdm(journals):
        sample_journal(journal)

data = get_sample_qa()

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