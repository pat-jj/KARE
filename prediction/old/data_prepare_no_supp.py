from spliter import split_by_patient
import pickle
import json


TASKS = {
    "mortality": {
        "description": """
Mortality prediction predicts the mortality label of the subsequent visit for patient where 1 indicates mortality and 0 indicates survival. 

Instructions for generating reasoning chains:
- Carefully examine the patient's conditions, procedures, medications across different visits (if any) to identify any high-risk factors or life-threatening situations.
- Consider the severity and progression of the patient's health issues over time.
- Assess the effectiveness of current treatments and management strategies in stabilizing the patient's condition.
- Identify any potential complications or deterioration in the patient's health that may lead to mortality in the subsequent visit.
- Provide a clear, step-by-step explanation of how the patient's information leads to the predicted mortality outcome.
"""
    },
    "readmission": {
        "description": """
Readmission prediction predicts if the patient will be readmitted into hospital within 15 days where 1 indicates readmission and 0 indicates no readmission.

Instructions for generating reasoning chains:
- Analyze the patient's recent visits and the reasons for hospitalization to identify any unresolved health issues or potential complications.
- Assess the patient's adherence to post-discharge instructions and follow-up care plans.
- Consider the patient's social determinants of health, such as support system, living conditions, and access to healthcare, which may impact their risk of readmission.
- Evaluate the effectiveness of current treatments and management strategies in preventing readmission.
- Provide a clear, step-by-step explanation of how the patient's information leads to the predicted readmission outcome within the specified time frame.
"""
    },
}

LABEL_MAPPING = {
    "mortality": {
        0: "0 (Survival in the subsequent visit)",
        1: "1 (Mortality in the subsequent visit)"
    },
    "readmission": {
        0: "0 (No Readmission within 15 days)",
        1: "1 (Readmission within 15 days)"
    },
}


def construct_input_output(patient_context, task, reasoning, ground_truth):
    
    context = patient_context.split("\n\nSupplementary Information:")[0]
    supplementary_info = patient_context.split("\n\nSupplementary Information:\n")[1:][0]
    
    input_ = f"""
Given the following patient information, supplementary information, prediction task, Please provide a step-by-step reasoning process that leads to the correct prediction based on the patient's context and relevant medical knowledge. The reasoning should be coherent, medically sound, and clearly explain how the patient's information leads to the predicted outcome:

# Patient Context #
{context}

========================================
# Task # 
{TASKS[task]["description"]}

"""

    output_ = f"""
# Reasoning #
{reasoning}

# Prediction #
{ground_truth}
"""

    return input_, output_



train_data_combined, val_data_combined, test_data_combined = [], [], []

for dataset in ["mimic3", 
                # "mimic4"
                ]:
    for task in ["mortality", "readmission"]:
        ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}.pkl"
        context_dir = f"/shared/eng/pj20/kelpie_exp_data/patient_context/augmented_context/patient_contexts_{dataset}_{task}.json"
        reference_dir = f"/shared/eng/pj20/kelpie_exp_data/reference_case/reference_cases_{dataset}_{task}.json"
        save_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_no_supp"
        
        
        patient_id_train, patient_id_val, patient_id_test = set(), set(), set()
        
        sample_dataset = pickle.load(open(ehr_path, "rb"))
        patient_contexts = json.load(open(context_dir, "r"))
        reference_cases = json.load(open(reference_dir, "r"))
        
        
        train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
        for sample in train_dataset:
            patient_id_train.add(sample["patient_id"])
        for sample in val_dataset:
            patient_id_val.add(sample["patient_id"])
        for sample in test_dataset:
            patient_id_test.add(sample["patient_id"])
            
        train_data, val_data, test_data = [], [], []
            
        for patient_id in reference_cases.keys():
            patient_context = patient_contexts[patient_id]
            reasoning = reference_cases[patient_id]["best chain"]
            ground_truth = reference_cases[patient_id]["ground_truth"]
            
            input_, output_ = construct_input_output(patient_context, task, reasoning, ground_truth)
            item = {"input": input_, "output": output_}
            
            if patient_id.split("_")[0] in patient_id_train:
                train_data.append(item)
                train_data_combined.append(item)
                
            elif patient_id.split("_")[0] in patient_id_val:
                val_data.append(item)
                val_data_combined.append(item)
                
            elif patient_id.split("_")[0] in patient_id_test:
                test_data.append(item)
                test_data_combined.append(item)
                
                
        with open(f"{save_path}/{dataset}_{task}_train.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
            for item in val_data:
                f.write(json.dumps(item) + "\n")
                
        with open(f"{save_path}/{dataset}_{task}_val.jsonl", "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")
                
        with open(f"{save_path}/{dataset}_{task}_test.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
                
                
with open(f"{save_path}/combined_train.jsonl", "w") as f:
    for item in train_data_combined:
        f.write(json.dumps(item) + "\n")
    for item in val_data_combined:
        f.write(json.dumps(item) + "\n")
        
with open(f"{save_path}/combined_val.jsonl", "w") as f:
    for item in val_data_combined:
        f.write(json.dumps(item) + "\n")
        
with open(f"{save_path}/combined_test.jsonl", "w") as f:
    for item in test_data_combined:
        f.write(json.dumps(item) + "\n")
            
        
                
                
        
        
    