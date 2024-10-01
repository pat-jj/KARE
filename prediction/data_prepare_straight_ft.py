from spliter import split_by_patient
import pickle
import json
import random
from tqdm import tqdm

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

TASKS_ABBR = {
    "mortality": {
        "description": """
Mortality Prediction Task:
Objective: Predict the mortality outcome for a patient's subsequent hospital visit.
Labels: 1 = mortality, 0 = survival
"""
    },
    "readmission": {
        "description": """
Readmission Prediction Task:
Objective: Predict if the patient will be readmitted to the hospital within 15 days of discharge.
Labels: 1 = readmission within 15 days, 0 = no readmission within 15 days
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

def construct_input_output(patient_context, task, ground_truth):
    
    context = patient_context
    
    input_ = f"""
Given the following task description and patient EHR context, please directly predict the label.

========================================
# Task #
{TASKS_ABBR[task]['description']}

========================================
# Patient EHR Context #

{context}
"""

    output_ = f"""{ground_truth}"""

    return input_, output_


for dataset in [
    "mimic3", 
                # "mimic4"
                ]:
    for task in [
        "mortality",
        "readmission"
                 ]:
        
        context_dir = f"/shared/eng/pj20/kelpie_exp_data/patient_context/base_context/patient_contexts_{dataset}_{task}.json"
        similar_patient_path = f"/shared/eng/pj20/kelpie_exp_data/patient_context/similar_patient/patient_to_top_1_patient_contexts_{dataset}_{task}.json"
        patient_data_path =  f"/shared/eng/pj20/kelpie_exp_data/ehr_data/pateint_{dataset}_{task}.json"
        test_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_test.json"
        train_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_train.json"
        val_sample_path = f"/shared/eng/pj20/kelpie_exp_data/ehr_data/{dataset}_{task}_samples_val.json"
        
        save_path = f"/shared/eng/pj20/kelpie_exp_data/llm_finetune_data_straight_ft"
        
        
        patient_contexts = json.load(open(context_dir, "r"))
        similar_patient = json.load(open(similar_patient_path, "r"))
        patient_data = json.load(open(patient_data_path, "r"))
        test_sample = json.load(open(test_sample_path, "r"))
        train_sample = json.load(open(train_sample_path, "r"))
        val_sample = json.load(open(val_sample_path, "r"))
        
        patient_id_test = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in test_sample]
        patient_id_train = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in train_sample]
        patient_id_val = [f"{sample['patient_id']}_{sample['visit_id']}" for sample in val_sample]
        patient_id_all = patient_id_train + patient_id_val + patient_id_test
        random.shuffle(patient_id_all)
        

            
        train_data, val_data, test_data = [], [], []
            
        for patient_id in tqdm(patient_id_all):
            patient_context = patient_contexts[patient_id]
            ground_truth = patient_data[patient_id]["label"]
            
            input_, output_ = construct_input_output(patient_context, task, ground_truth)
            item = {"input": input_, "output": output_}
            
            # print(f"Patient ID: {patient_id}, Input: {input_}, Output: {output_}")
            
            if patient_id in patient_id_train:
                train_data.append(item)
                
            elif patient_id in patient_id_val:
                val_data.append(item)
                
            elif patient_id in patient_id_test:
                test_data.append(item)
                
                
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
                
                
        
                
                
        
        
    