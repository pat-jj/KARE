import os
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from flask import Flask, request, jsonify


os.environ['CUDA_VISIBLE_DEVICES'] = '7'

# define logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)



# define server app
app = Flask(__name__)


# load model, tokenizer
trained_model_path = '/shared/eng/pj20/kelpie_exp_data/llm_training_output/combined/mistral-7b-sft-full'
# trained_model_path = '/shared/eng/langcao2/LEADS_llm/training_output/mistral-7b-sft-full-search'


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
model = AutoModelForCausalLM.from_pretrained(
    trained_model_path,
    device_map='auto',
    torch_dtype=torch.bfloat16,
    attn_implementation='flash_attention_2'
)

# define terminators
terminators = [tokenizer.eos_token_id]
if 'llama' in trained_model_path:
    llama3_stop_tokens = ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>']
    terminators.extend([tokenizer.convert_tokens_to_ids(T) for T in llama3_stop_tokens])
elif 'mistral' in trained_model_path:
    mistral_stop_tokens = ['[INST]', '[/INST]']
    terminators.extend([tokenizer.convert_tokens_to_ids(T) for T in mistral_stop_tokens])



# generate api
@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    prompt = data.get('prompt', '')
    max_length = data.get('max_length', 30_000)
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
    logger.info(f'Input prompt: {formatted_prompt}')

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        temperature=temperature,
        do_sample=do_sample,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=terminators,
        pad_token_id = tokenizer.eos_token_id
    )

    generated_tokens = outputs[0, inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    logger.info(f'Generated response: {response}')

    return jsonify({'response': response})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1212)