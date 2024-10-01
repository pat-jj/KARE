import json

dataset = "mimic3"
task = "mortality"
output_path = "/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_multitask"

        
# ==========================================
# train data
ori_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_ulti/{dataset}_{task}_train.jsonl"

with open(ori_path, "r") as f:
    data = [json.loads(line) for line in f]
    
    
instruction_prev = "\nGiven the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.\nAfter the reasoning process, provide the prediction label (0/1)."
instruction_new_reason = "\n[Reasoning] Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.\nAfter the reasoning process, provide the prediction label (0/1)."
instruction_new_pred = "\n[Label Prediction] Given the following task description, patient EHR context, similar patients, and retrieved medical knowledge, Please directly predict the label (0/1).\n"


label_pred_data = []
reasoning_data = []

for item in data:
    input_new = item["input"].replace(instruction_prev, instruction_new_pred)
    output_new = item["output"][-1]
    label_pred_data.append({"input": input_new, "output": output_new})
    
    input_new = item["input"].replace(instruction_prev, instruction_new_reason)
    output_new = item["output"]
    reasoning_data.append({"input": input_new, "output": output_new})
    
data = label_pred_data + reasoning_data
with open(f"{output_path}/{dataset}_{task}_train_multitask.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
        
        
# ==========================================
# test data
ori_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_ulti/{dataset}_{task}_test.jsonl"

with open(ori_path, "r") as f:
    data = [json.loads(line) for line in f]
    
    
label_pred_data = []

for item in data:
    input_new = item["input"].replace(instruction_prev, instruction_new_pred)
    output_new = item["output"][-1]
    label_pred_data.append({"input": input_new, "output": output_new})
    
with open(f"{output_path}/{dataset}_{task}_test_multitask.jsonl", "w") as f:
    for item in label_pred_data:
        f.write(json.dumps(item) + "\n")