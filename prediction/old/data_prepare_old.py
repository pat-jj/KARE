from spliter import split_by_patient
import pickle
import json
from tqdm import tqdm
import os

REF_IDS = [3]

TASKS = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit based solely on conditions, procedures, and medications. 
Labels: 1 = mortality, 0 = survival

Key Considerations:
1. Conditions:
   - Severity of diagnosed conditions (e.g., advanced cancer, severe heart failure, sepsis)
   - Presence of multiple comorbidities
   - Acute vs. chronic nature of conditions

2. Procedures:
   - Invasiveness and complexity of recent procedures 
   - Emergency vs. elective procedures
   - Frequency of life-sustaining procedures (e.g., dialysis, mechanical ventilation)

3. Medications:
   - Use of high-risk medications (e.g., chemotherapy drugs, immunosuppressants)
   - Multiple medication use indicating complex health issues
   - Presence of medications typically used in end-of-life care

Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality (definitely die) should be predicted as 1.

"""
    },
    "readmission": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge based solely on conditions, procedures, and medications.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days

Key Considerations:  
1. Conditions:
   - Chronic diseases with high risk of exacerbation (e.g., COPD, heart failure)
   - Conditions requiring close monitoring or frequent adjustments (e.g., diabetes)
   - Recent acute conditions with potential for complications

2. Procedures:
   - Recent major surgeries or interventions with high complication rates
   - Procedures that require extensive follow-up care
   - Incomplete or partially successful procedures

3. Medications:  
   - New medication regimens that may require adjustment
   - Medications with narrow therapeutic windows or high risk of side effects
   - Complex medication schedules that may lead to adherence issues

Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.

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
Given the following task description, patient information, supplementary information, Please provide a step-by-step reasoning process that leads to the prediction outcome based on the patient's context and relevant medical knowledge.
After the reasoning process, provide the prediction label (0/1) based on the reasoning process.

========================================
# Task #
{TASKS[task]["description"]}

========================================
# Patient Context #

{context}

========================================
# Supplementary Information #

{supplementary_info}



"""

    output_ = f"""
# Reasoning #
{reasoning}

# Prediction #
{ground_truth[0]}"""

    # print(input_)
    # print(output_)

    return input_, output_
        

train_data_combined, val_data_combined, test_data_combined = [], [], []

for dataset in ["mimic3", 
                # "mimic4"
                ]:
    for task in [
        "mortality", 
        "readmission"
        ]:
        ehr_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}.pkl"
        context_dir = f"/shared/eng/pj20/kelpie_exp_data/patient_context/augmented_context/patient_contexts_{dataset}_{task}.json"
        save_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data"
        
        
        patient_id_train, patient_id_val, patient_id_test = set(), set(), set()
        
        sample_dataset = pickle.load(open(ehr_path, "rb"))
        patient_contexts = json.load(open(context_dir, "r"))
        
        
        train_dataset, val_dataset, test_dataset = split_by_patient(sample_dataset, [0.8, 0.1, 0.1], seed=528)
        for sample in train_dataset:
            patient_id_train.add(sample["patient_id"])
        for sample in val_dataset:
            patient_id_val.add(sample["patient_id"])
        for sample in test_dataset:
            patient_id_test.add(sample["patient_id"])
            
        train_data, val_data, test_data = [], [], []
        
        for ref_id in REF_IDS:
            reference_dir = f"/shared/eng/pj20/kelpie_exp_data/reference_case/reference_cases_{dataset}_{task}_{ref_id}.json"
            if os.path.exists(reference_dir):
                reference_cases = json.load(open(reference_dir, "r"))
                
                for patient_id in tqdm(reference_cases.keys()):
                    patient_context = patient_contexts[patient_id]
                    reasoning = reference_cases[patient_id]["best chain"]
                    ground_truth = reference_cases[patient_id]["ground_truth"]
                    if (patient_id.split("_")[0] in patient_id_train or patient_id.split("_")[0] in patient_id_val) and "\n# Confidence #\nNot Confident" in reasoning:
                        continue
                    reasoning = reasoning.split("\n# Confidence #\n")[0]
                    reasoning = reasoning.replace("# Reasoning Chain #\n", "")
                    
                    input_, output_ = construct_input_output(patient_context, task, reasoning, ground_truth)
                    item = {"input": input_, "output": output_}
                    
                    if patient_id.split("_")[0] in patient_id_train:
                        train_data.append(item)
                        train_data_combined.append(item)
                        
                    if ref_id == 0:
                        
                        if patient_id.split("_")[0] in patient_id_val:
                            val_data.append(item)
                            val_data_combined.append(item)
                            
                        elif patient_id.split("_")[0] in patient_id_test:
                            test_data.append(item)
                            test_data_combined.append(item)
                        
                        
        with open(f"{save_path}/{dataset}_{task}_train_{REF_IDS[0]}.jsonl", "w") as f:
            for item in train_data:
                f.write(json.dumps(item) + "\n")
            for item in val_data:
                f.write(json.dumps(item) + "\n")
                
                
        with open(f"{save_path}/{dataset}_{task}_val_{REF_IDS[0]}.jsonl", "w") as f:
            for item in val_data:
                f.write(json.dumps(item) + "\n")
                
        with open(f"{save_path}/{dataset}_{task}_test_{REF_IDS[0]}.jsonl", "w") as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
                
                
# with open(f"{save_path}/combined_train.jsonl", "w") as f:
#     for item in train_data_combined:
#         f.write(json.dumps(item) + "\n")
#     for item in val_data_combined:
#         f.write(json.dumps(item) + "\n")
        
# with open(f"{save_path}/combined_val.jsonl", "w") as f:
#     for item in val_data_combined:
#         f.write(json.dumps(item) + "\n")
        
# with open(f"{save_path}/combined_test.jsonl", "w") as f:
#     for item in test_data_combined:
#         f.write(json.dumps(item) + "\n")
            
        
                
                
        
        
    