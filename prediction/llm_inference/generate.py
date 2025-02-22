import json
from collections import defaultdict
from tqdm import tqdm
import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import requests
import threading
import io
import sys


# ================== CONFIGURATION ==================
DATASET = "mimic3"
# TASK = "mortality"
TASK = "readmission"
MODEL = ''
# TYPE = '-adpt-weighted-0.33'
TYPE='knowledge_dp'
# MODEL = '_no_supp'
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


PORT = 3343
# MODEL_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_training_output/mortality_mimic4_multitask_0/checkpoint-544"
MODEL_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_training_output/readmission_mimic4_multitask_0/checkpoint-888"
# TEST_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_multitask/mimic4_mortality_test_multitask_reason.jsonl"
# TEST_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_multitask/mimic4_readmission_test_multitask_reason.jsonl"
TEST_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_multitask/mimic4_readmission_test_multitask.jsonl"

# SAVE_PATH = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_multitask_reason.json"
SAVE_PATH = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_multitask_speed.json"

# PORT = 3333
# MODEL_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_training_output/readmission_mimic4_straight_ft/checkpoint-108"
# TEST_PATH = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_straight_ft/mimic4_readmission_test_filtered.jsonl"
# SAVE_PATH = "/shared/eng/pj20/kelpie_exp_data/results/mimic4_readmission_straight_ft_filtered.json"

MAX_LENGTH = 6000


# ===================================================


# Redirect Flask and model output
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = io.StringIO()

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.logger.disabled = True
log = logging.getLogger('werkzeug')
log.disabled = True

# Load model and tokenizer
trained_model_path = MODEL_PATH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForCausalLM.from_pretrained(
    trained_model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2',
)

# Define terminators
terminators = [tokenizer.eos_token_id]
if 'llama' in trained_model_path:
    llama3_stop_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
    terminators.extend([tokenizer.convert_tokens_to_ids(T) for T in llama3_stop_tokens])
elif 'mistral' in trained_model_path:
    mistral_stop_tokens = ['[INST]', '[/INST]']
    terminators.extend([tokenizer.convert_tokens_to_ids(T) for T in mistral_stop_tokens])

# Optimize model for inference
model.eval()

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_length = data.get('max_length', 6000)
        temperature = data.get('temperature', 1.0)
        do_sample = data.get('do_sample', False)
        top_k = data.get('top_k', 50)
        top_p = data.get('top_p', 1.0)
        
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                do_sample=do_sample,
                top_k=top_k,
                top_p=top_p,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id
            )


        generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    except Exception as e:
        logger.error(f'Error: {e}')

    return jsonify({'response': response})

def get_llm_response(prompt, max_length=1000, temperature=1.0):
    url = f"http://localhost:{PORT}/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "prompt": prompt,
        "max_length": max_length,
        "temperature": temperature,
    }

    with HiddenPrints():
        response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('response', 'No response key in JSON')
    else:
        return f"Error: {response.status_code}, {response.text}"

def run():    
    test_file_path = TEST_PATH
    results_file_path = SAVE_PATH
    result = defaultdict(dict)
    
    with open(test_file_path, "r") as f:
        data = f.readlines()
    
    data = [json.loads(line) for line in data]
    
    for idx, sample in enumerate(tqdm(data, desc="Processing samples", ncols=100)):
        input_ = sample["input"]
        reasoning_and_prediction = get_llm_response(input_, max_length=MAX_LENGTH)
        ground_truth = sample["output"]
        
        result[idx]["input"] = input_
        result[idx]["ground_truth"] = ground_truth
        result[idx]["reasoning_and_prediction"] = reasoning_and_prediction
        
        with open(results_file_path, "w") as f:
            json.dump(result, f, indent=4)
        
    with open(results_file_path, "w") as f:
        json.dump(result, f, indent=4)

def start_flask():
    app.run(host='0.0.0.0', port=PORT, debug=False)

if __name__ == '__main__':
    flask_thread = threading.Thread(target=start_flask)
    flask_thread.start()

    run()

    flask_thread.join()