import json
import torch
import re
import os
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
from transformers import AutoTokenizer, AutoModelForCausalLM

from .eval import input_file_path

llm_output_file_path = 'results/llm_evaluation.json'

model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_name, token=os.environ.get('token'))
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", token=os.environ.get('token'))

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
special_tokens_dict = {"pad_token": "<pad>", "eos_token": "</s>"}
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)

SYS_PROMPT = "You are a fair evaluator language model."

def extract_score_from_text(text):
    """
    Extracts the score after '[RESULT]' in the given text.
    
    Parameters:
    text (str): The input text containing '[RESULT]' and a score.

    Returns:
    int: The extracted score as an integer, or None if no score is found.
    """
    result_number = re.search(r'\[RESULT\]\s*(\d+)', text)
    if result_number:
        return int(result_number.group(1)) 
    else:
        return text


def generate(formatted_prompt):
    messages = [{"role":"system","content":SYS_PROMPT},{"role":"user","content":formatted_prompt}]
    
    input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids,
        max_new_tokens=500,
        eos_token_id = terminators,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=False,
        temperature = None,
        top_p = None,
        top_k = None
    )
    response = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(response, skip_special_tokens=True)

def format_query_prompt(query, gt, pd):
    PROMPT = f"""
### Task Description:
1. Determine whether the predicted answer provides the similar level of detail and explanation as the reference answer (ground truth answer).
2. If the reference answer includes specific steps, procedures, or details, the predicted answer must also include similarly specific information. 
3. General or vague responses that do not cover the key details will be considered incorrect, output "0".
4. If the predicted answer only briefly mentions the topic without providing the necessary level of detail or fails to explain the key points, output "0".
5. If the predicted answer is concise but covers all critical steps or points provided in the reference answer, output "1".
6. Never generate any other content besides the binary score.
7. The output format should be strictly: {{either 0 or 1}}.

### Question : 
{query}

### Predicted:
{pd}

### Reference:
{gt}

### Evaluation Criteria:
1. If the predicted answer provides a similar level of detail, covers the key points, and aligns with the context of the query and the reference answer, output 1.
2. If the predicted answer is concise but accurately captures the core concept or key term(s) from the reference answer in response to the query, output 1.
3. If the predicted answer is too general, omits important details, lacks critical steps, output 0.
4. If the predicted answer does not share key terms or concepts with the reference answer or deviates significantly from the context, output 0.
5. If the predicted answer is empty(e.g "{{}}"), output 0.

"""
    return PROMPT



def RAG_eval_w_LLM(eval_path, output_path, verbose=False):
    total_score = 0
    error = 0
    
    with open(output_path, 'w', encoding='utf-8') as output_file:
        json.dump([], output_file)  

    with open(eval_path, 'r', encoding='utf-8') as eval_file:
        for idx, line in tqdm(enumerate(eval_file)):
            try: 
                dic = json.loads(line)
                question = dic['question']
                groundtruth = dic['answers']
                prediction = dic['generated_answer']
                prompt = format_query_prompt(question, groundtruth, prediction)
                generated_eval = generate(prompt)
                score = extract_score_from_text(generated_eval)
            except:
                error += 1
                print(f'Error # {error}')
                score = 0
            try:
                total_score += int(score)
            except:
                total_score += 0

            result_dict = {
                "index": idx + 1,
                "question": question,
                "ground_truth": groundtruth,  
                "generated_answer": prediction,
                "score": score,
                "current_avg_score": round(total_score / (idx + 1), 4) 
            }
            
            with open(output_path, 'r', encoding='utf-8') as output_file:
                existing_data = json.load(output_file)
            
            existing_data.append(result_dict)
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(existing_data, output_file, ensure_ascii=False, indent=4)


            if verbose:
                print('=' * 50)
                print(f'# {idx + 1} Processing')
                print(f'Question      : {question}')
                print(f'Ground Truth  : {groundtruth}')
                print(f'Generated Answer: {prediction}')
                print(f'Generated Eval: {generated_eval}')
                print(f'Score         : {score}')
                print(f'Current Avg Score: {total_score / (idx + 1):.4f}')
                print('=' * 50 + '\n')
        
        final_avg_score = total_score / (idx + 1)
        final_result = {"final_avg_score": round(final_avg_score, 4)}

    print(f'Final Avg Score: {final_avg_score:.2f}')
    return final_avg_score

     
if __name__ == '__main__':
    RAG_eval_w_LLM(input_file_path, llm_output_file_path)
